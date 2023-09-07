import cv2
import numpy as np
from ultralytics import YOLO
import sympy
from warnings import warn

model = YOLO('runs/segment/yolov8n_azul15/weights/best.pt')
# sz=2800x2800, border=150
PATH_TEMPLATE = 'assets/templates/board.png'
_template = cv2.imread(PATH_TEMPLATE)

RECT_SCORE = (314-150, 219-150, 2478-314, 914-219)
RECT_LINE = (195, 989, 1230-344, 2036-1139)
RECT_GRID = (1556-150, 1139-150, 2434-1556, 2036-1139)
RECT_FLOOR = (195, 2258, 1783-344, 198)

SIZE_TILE = (198, 198)
SIZE_MARKER = (215-112, 255-125)

SHAPE_SCORE = (6, 20)
SHAPE_LINE = (5, 5)
SHAPE_GRID = (5, 5)
SHAPE_FLOOR = (2, 7)


def get_boards(img, plot=False):
    imgClean = img.copy()
    masks = segment(img, plot)
    maskMerge = np.sum(np.uint8(masks), axis=0).astype(np.uint8)

    h1, w1 = masks[0].shape
    hf, wf, _ = img.shape
    BRD, FSZ = 50, 800
    dst = np.float32([[BRD, BRD], [FSZ-BRD, BRD],
                     [FSZ-BRD, FSZ-BRD], [BRD, FSZ-BRD]])

    BRD2, FSZ2 = 0, 640
    sz = FSZ2 / 640
    dst2 = np.float32([[BRD2, BRD2], [FSZ2-BRD2, BRD2],
                      [FSZ2-BRD2, FSZ2-BRD2], [BRD2, FSZ2-BRD2]])

    MIN_MATCH_VAL = 1500
    BORDER_COLOR = (81, 56, 40)

    imgBoard = []
    for i, data in enumerate(masks):
        mask = np.uint8(data)
        src = corners_from_mask(mask)
        matrix = cv2.getPerspectiveTransform(
            np.float32(src * [wf/w1, hf/h1]), dst)
        frame = cv2.warpPerspective(
            imgClean, matrix, (FSZ, FSZ), borderMode=cv2.BORDER_REPLICATE)

        src2 = corner_from_closeup(frame)[0]
        if src2 is None:
            warn('Missing edge from board')
            continue

        matrix2 = cv2.getPerspectiveTransform(np.float32(src2), dst2)
        frame = cv2.warpPerspective(
            imgClean, matrix2 @ matrix, (FSZ2, FSZ2), borderValue=BORDER_COLOR)

        frame, valMatch = orient_board(frame, border=FSZ2//50)
        if valMatch < MIN_MATCH_VAL:
            warn('Bad match')
            continue
        if plot:
            # M = matrix2 @ matrix
            # Psrc2 = [np.linalg.solve(M[:2, :2], pt - M[:2, 2]) for pt in src2]

            # cv2.polylines(img, [np.int32(Psrc2)],
            #               True, color=(0, 0, 255), thickness=int(np.ceil(5*sz)))

            cv2.polylines(maskMerge, [np.int32(src)],
                          True, color=2, thickness=int(np.ceil(1*sz)))
            for pt in src:
                cv2.circle(maskMerge, np.int32(pt), int(
                    3*sz), color=2, thickness=-1)
            cv2.putText(frame, f"{int(valMatch)}", (int(10*sz), int(40*sz)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2*sz, (50, 50, 255), int(3*sz))

        imgBoard.append(frame)

    return imgBoard, maskMerge


def segment(img, plot=False):
    results = model.predict(source=img, save=False, save_txt=False, conf=0.75)

    if plot:
        sz = 2 * img.shape[0] / 640
        class_ids, confidences, boxes = [], [], []
        for (x, y, x2, y2, conf, id) in results[0].boxes.data:
            class_ids.append(int(id))
            confidences.append(float(conf))
            boxes.append(np.int32([x, y, x2-x, y2-y]))

        colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 0)]

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(img, box, color, int(2*sz))
            cv2.rectangle(img, (box[0], box[1] - int(20*sz)),
                          (box[0] + box[2], box[1]), color, -1)
            cv2.putText(img, f"{model.names[classid]} {confidence:.2}",
                        (box[0], box[1] - int(3*sz)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*sz, (255, 255, 255), int(2*sz))

    return results[0].masks.data


def signed_area(u, v):
    return u[0] * v[1] - u[1] * v[0]


# TODO test with polygons besides N=4
def corners_from_mask(mask, numCorners=4):
    # [s0, v] * N
    # s0: vertex
    # v: direction vector of edge, oriented toward +z
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get largest blob
    extent = [xy[:, 0, 0].max() - xy[:, 0, 0].max() +
              xy[:, 0, 1].max() - xy[:, 0, 1].min() for xy in contours]
    xy = contours[np.argmax(extent)][:, 0, :]

    # resample points to make consecutive points equally distanced
    xyup = np.vstack([xy, xy[0]])
    clen = np.hstack(
        [0, np.cumsum(np.sqrt(np.sum((xyup[1:] - xyup[:-1])**2, 1)))])
    cclen = np.linspace(0, clen[-1], 100)
    xy = np.array([np.interp(cclen, clen, xyup[:, i]) for i in range(2)]).T

    # cluster points on N edges. Use slope and cyclic sequence as features
    seq = np.arange(len(xy))
    seqSin = np.sin(seq * 2*np.pi / seq[-1])
    seqCos = np.cos(seq * 2*np.pi / seq[-1])
    dxy = (np.gradient(xy, 3)[0] * 3)
    features = np.concatenate(
        [dxy, seqSin[:, np.newaxis], seqCos[:, np.newaxis]], axis=1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    labels = None
    retval, labels, _ = cv2.kmeans(features, numCorners,
                                   labels, criteria, 10, flags)

    # calc line coefficients as edge center and unit direction
    coefs = []
    for i in range(numCorners):
        xy0 = xy[labels.flatten() == i][2:-2]
        s0 = np.mean(xy0, 0)
        v = np.linalg.svd(xy0 - s0)[2][0, :]
        # cv2.polylines(mask, [np.int32(xy0)], False, 2, 2)
        # cv2.polylines(mask, [np.int32(np.vstack([s0, s0 + v * 50]))], False, 3, 2)
        coefs.append([s0, v])

    return corners_from_coefs(coefs)


def corner_from_closeup(frame):
    t1, ksz, asz = 40, 7, 3
    brd = 100
    minVotes = 75
    stepRho = 0.5
    stepTheta = 0.25 * np.pi/180
    h, w = frame.shape[:2]

    coefs = []
    imgs = []
    for j in range(4):
        matrix = cv2.getRotationMatrix2D([w//2, h//2], j*90, scale=1)
        matrixInv = cv2.invertAffineTransform(matrix)
        src = cv2.warpAffine(frame, matrix, [w, brd])

        dst = cv2.blur(src, (ksz, ksz))
        dst = cv2.blur(dst, (ksz, ksz))
        dst = cv2.Canny(dst, threshold1=t1, threshold2=t1 *
                        2, edges=None, apertureSize=asz)
        # dst[:5, :] = 0
        imgs.append(dst)
        lines = cv2.HoughLinesP(dst, rho=stepRho, theta=stepTheta,
                                threshold=minVotes, minLineLength=100, maxLineGap=200)

        if lines is None:
            return None, imgs
        else:
            linesB = lines[:5, 0, :]  # best lines
            y = linesB[:, 1] + linesB[:, 3]
            theta = np.arctan2(
                linesB[:, 1] - linesB[:, 3], linesB[:, 0] - linesB[:, 2])

            # best line is the top-most out of most-voted 5
            # idxBest = np.argmin(y)

            # best line is the top-most out of most-voted 5 that is mostly vertical
            idxBest = np.argmin(y + 1_000*(np.abs(theta) < 5*np.pi/180))

            line = lines[idxBest, 0, :]

            p1, p2 = line[:2], line[2:]
            P1 = np.int32(matrixInv @ np.r_[p1, 1])
            P2 = np.int32(matrixInv @ np.r_[p2, 1])
            s0 = (P1 + P2) / 2
            v = (P2 - P1) / np.sqrt(((P2 - P1)**2).sum())
            coefs.append([s0, v])

    # return coefs
    return corners_from_coefs(coefs), imgs  # , coefs


def corners_from_coefs(coefs, numCorners=4):
    # calc intersection of edges, filtering out non intersecting edges
    c0 = np.mean([s0 for s0, v in coefs], 0)
    corners = np.ones([numCorners, 2]) * np.Inf
    idxPrev = [0] * numCorners
    for i, (si, vi) in enumerate(coefs):
        for j in range(i+1, numCorners):
            sj, vj = coefs[j]

            try:
                ts = np.linalg.solve(np.c_[vi, -vj], sj - si)
            except np.linalg.LinAlgError:
                # probably v pointing to same direction
                continue

            pt = si + ts[0] * vi

            # color = i*numCorners + j + 2
            # pts = np.c_[si, pt, sj].T
            # cv2.polylines(mask, [np.int32(pts)], False, color, thickness=2)
            # cv2.circle(mask, np.int32(pt), 10, color, thickness=3)

            idx, idxN = (i, j) if signed_area(si - c0, sj - c0) < 0 else (j, i)
            if np.sum((pt - c0)**2) < np.sum((corners[idx, :] - c0)**2):
                corners[idx, :] = pt
                idxPrev[idx] = idxN
                if coefs[idx][1] @ (coefs[idx][0] - pt) < 0:
                    coefs[idx][1] *= -1

    # sort corners in +z order
    corners_ = np.zeros([numCorners, 2])
    coefs_ = np.zeros([numCorners, 4])
    idx = idxPrev[-1]
    for i in range(numCorners-1, -1, -1):
        corners_[i] = corners[idx]
        coefs_[i] = np.r_[coefs[idx][0], coefs[idx][1]]
        idx = idxPrev[idx]
        # cv2.polylines(mask, [np.int32(pts)], False, 3, thickness=2)
        # cv2.putText(mask, f"{j}", np.int32(pt), 1, fontScale=2, color=5, thickness=2)

    # make first index the top-most
    idx = np.argmin(corners_[:, 1])  # top-most
    corners_ = np.roll(corners_, -idx, axis=0)
    coefs_ = np.roll(coefs_, -idx, axis=0)

    # for j in range(numCorners):
    #     sj, vj = coefs_[j][:2], coefs_[j][2:]
    #     pt = corners_[j]
    #     pts = np.c_[pt, pt + vj * 150].T
    #     cv2.polylines(mask, [np.int32(pts)], False, 2, thickness=1)
    #     cv2.putText(mask, f"{j}", np.int32(pt), 1, fontScale=2, color=3, thickness=2)
    return corners_  # , (idxNext, xy, labels, np.c_[dxy, seqSin, seqCos])


def orient_board(imgBoard, border=0):
    h, w = imgBoard.shape[:2]
    TPT_SHAPE = (w + 2*border, h + 2*border)
    template = cv2.resize(_template, TPT_SHAPE)

    bestVal = -np.Inf
    img = imgBoard.copy()
    for i in range(4):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        val = res.max()
        # print(f"{val=}")
        if val > bestVal:
            bestVal = val
            imgBoard = img.copy()

    return imgBoard, bestVal / (TPT_SHAPE[0] * TPT_SHAPE[1])


def get_tile_features(img):
    """
        img: in BGR space
    """
    histB = cv2.calcHist([img], [0], None, histSize=4, ranges=(0, 256))
    print(histB)
    pass


def appx_best_fit_ngon(mask_cv2_gray, n: int = 4):
    # convex hull of the input mask
    # mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask_cv2_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(
                edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull
