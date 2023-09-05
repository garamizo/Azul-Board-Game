import cv2
import numpy as np
from ultralytics import YOLO
import sympy

model = YOLO('runs/segment/yolov8n_azul15/weights/best.pt')


def segment(img):
    results = model.predict(source=img, save=False, save_txt=False, conf=0.5)
    return results[0].masks.data


def signed_area(u, v):
    return u[0] * v[1] - u[1] * v[0]


# TODO test with polygons besides N=4
def poly_corners(mask, numCorners=4):
    # [s0, v] * N
    # s0: vertex
    # v: direction vector of edge, oriented toward +z
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xy = contours[0][:, 0, :]

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
    _, labels, _ = cv2.kmeans(features, numCorners,
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

    # calc intersection of edges, filtering out non intersecting edges
    c0 = np.mean(xy, 0)
    corners = np.ones([numCorners, 2]) * np.Inf
    idxPrev = [0] * numCorners
    for i, (si, vi) in enumerate(coefs):
        for j in range(i+1, numCorners):
            sj, vj = coefs[j]

            ts = np.linalg.solve(np.c_[vi, -vj], sj - si)
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
