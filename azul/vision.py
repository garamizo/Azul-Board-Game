import torch
from torchvision import transforms
from tileclassifier.model import build_model
import cv2
import numpy as np
from ultralytics import YOLO
import sympy
from warnings import warn
import matplotlib.pyplot as plt

MODEL_PATH = '/home/garamizo/Azul-Board-Game/outputs/model_pretrained_True.pth'
model = YOLO('/home/garamizo/Azul-Board-Game/runs/segment/yolov8n_azul15/weights/best.pt')
# sz=2500x2500, border=0
PATH_TEMPLATE = '/home/garamizo/Azul-Board-Game/assets/templates/board.png'
_template = cv2.imread(PATH_TEMPLATE)

RECT_SCORE = [v/2500 for v in (314-150, 219-150, 2478-314, 914-219)]
RECT_LINE = [v/2500 for v in (195, 989, 1230-344, 2036-1139)]
RECT_GRID = [v/2500 for v in (1556-150, 1139-150, 2434-1556, 2036-1139)]
RECT_FLOOR = [v/2500 for v in (195, 2258, 1783-344, 198)]

SIZE_TILE = [v/2500 for v in (198, 198)]
SIZE_MARKER = [v/2500 for v in (215-112, 255-125)]

SHAPE_SCORE = (6, 20)
SHAPE_LINE = (5, 5)
SHAPE_GRID = (5, 5)
SHAPE_FLOOR = (2, 7)

orb = cv2.ORB_create(
    nfeatures=2_000,
    scaleFactor=1.2,
    edgeThreshold=15,
    patchSize=31,
    scoreType=cv2.ORB_HARRIS_SCORE)
orbLarge = cv2.ORB_create(
    nfeatures=2_000,
    scaleFactor=1.2,
    edgeThreshold=15,
    patchSize=31,
    scoreType=cv2.ORB_HARRIS_SCORE)
index_params = dict(
    algorithm=6,  # FLANN_INDEX_LSH
    table_number=6,
    key_size=10,
    multi_probe_level=2)


class FeatureExtraction:
    def __init__(self, img, isTemplate=False):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if isTemplate:
            self.kps, self.des = orbLarge.detectAndCompute(
                gray_img, None)
        else:
            self.kps, self.des = orb.detectAndCompute(
                gray_img, None)
        self.matched_pts = []


LOWES_RATIO = 0.7
MIN_MATCHES = 15
index_params = dict(
    algorithm=6,  # FLANN_INDEX_LSH
    table_number=6,
    key_size=10,
    multi_probe_level=2)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(
    index_params,
    search_params)


w, h, brd = 800, 800, 80
bVal = (185, 197, 203)
matrix = np.float32([
    [(w-2*brd)/2500, 0, brd],
    [0, (h-2*brd)/2500, brd]])
template = cv2.warpAffine(_template, matrix, [w, h], borderValue=bVal)
_features0 = FeatureExtraction(template, isTemplate=True)
_template0 = cv2.resize(_template, [640, 640])


def crop(img, rect):
    return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]


def tile(imgs, shape=None, grid=0):
    scale = True
    if shape == None:
        shape = imgs[0].shape
        scale = False

    numImgs = len(imgs)
    nx = int(np.ceil(np.sqrt(numImgs)))
    ny = int(np.ceil(numImgs / nx))
    if scale:
        imgs = [cv2.resize(img, shape[:2]) for img in imgs]
    imgs = imgs + [np.zeros(shape, np.uint8)] * (nx*ny - numImgs)
    tiles = np.reshape(imgs, [ny, nx, *shape])

    # for i in range(tiles.shape[3]):
    tiles[:, :, :grid, :, :] = 255
    tiles[:, :, :, :grid, :] = 255
    tiles[:, :, shape[0]-grid:, :, :] = 255
    tiles[:, :, :, shape[1]-grid:, :] = 255

    tiles = tiles.swapaxes(1, 2).reshape(shape[0]*ny, shape[1]*nx, shape[2])
    return tiles


class BoardDetector_YOLO:
    PATH_MODEL = 'runs/segment/yolov8n_azul15/weights/best.pt'
    PRED_CONF = 0.75  # only accept detections with confidence > 0.75
    PAD_BORDER = 0
    # shape of segmentation mask (fixed to YOLOv8-seg model)
    SHAPE_OUT = (1000, 1000)  # shape of output boards

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.model = YOLO(self.PATH_MODEL)

    def get_matrix(self, img, plot=False):
        if plot:
            self.imgDraw = img.copy()

        h, w, c = img.shape
        masks, boxes, confs = self.segment(img)

        dst = np.float32([
            [self.PAD_BORDER, self.PAD_BORDER],
            [self.SHAPE_OUT[1]-self.PAD_BORDER, self.PAD_BORDER],
            [self.SHAPE_OUT[1]-self.PAD_BORDER, self.SHAPE_OUT[0]-self.PAD_BORDER],
            [self.PAD_BORDER, self.SHAPE_OUT[0]-self.PAD_BORDER]
        ])
        for i, (mask, box, conf) in enumerate(zip(masks, boxes, confs)):
            shape_mask = mask.shape
            corners = polygon_from_mask(mask, numCorners=4) * \
                [w / shape_mask[1], h / shape_mask[0]]

            # apply perspective transf given polygon corners
            matrix = cv2.getPerspectiveTransform(np.float32(corners), dst)

            if plot:
                cv2.putText(self.imgDraw, f"{i}", tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX,
                            8, (0, 0, 255), 20)

                maskUpsample = cv2.resize(
                    mask, img.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                self.imgDraw[maskUpsample, 0] = 255
                cv2.polylines(self.imgDraw, [np.int32(corners)],
                              True, color=(0, 0, 255), thickness=10)
            yield matrix

    def detect(self, img, plot=False):
        imgs = []
        for matrix in self.get_matrix(img, plot):
            imgs.append(cv2.warpPerspective(
                img, matrix, self.SHAPE_OUT, borderMode=cv2.BORDER_REPLICATE))
        return imgs

    def segment(self, img, plot=False):
        """Compute masks and bounding boxes of board elements
            Returns:
                - masks: list of N masks of each board element
                - boxes: list of N bounding boxes of each board element
                - confidence: list of N confidence of each board element
        """
        results = self.model.predict(
            source=img, save=False, save_txt=False, conf=self.PRED_CONF, verbose=False)

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

        return [np.uint8(m) for m in results[0].masks.data], \
            [np.int32(np.round([b[0], b[1], b[2]-b[0], b[3]-b[1]])) for b in results[0].boxes.data], \
            [float(v[4]) for v in results[0].boxes.data]


class BoardDetector_ORB:
    PATH_TEMPLATE = 'assets/templates/board.png'
    SHAPE_OUT = (1000, 1000)
    MIN_MATCHES = 15
    PATCH_SIZE = 51
    N_FEATURES = 2_000
    index_params = dict(
        algorithm=6,  # FLANN_INDEX_LSH
        table_number=6,
        key_size=10,
        multi_probe_level=2)
    search_params = dict(checks=50)
    COLOR_BACKGROUND = (185, 197, 203)

    @property
    def PAD_BORDER(self): return 2 * self.PATCH_SIZE

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # init feature extractor ============================
        # tested with 1000x1000
        self.orb = cv2.ORB_create(
            nfeatures=self.N_FEATURES,
            edgeThreshold=self.PATCH_SIZE,
            patchSize=self.PATCH_SIZE,
            scoreType=cv2.ORB_HARRIS_SCORE)
        orbL = cv2.ORB_create(
            nfeatures=2*self.N_FEATURES,
            edgeThreshold=self.PATCH_SIZE,
            patchSize=self.PATCH_SIZE,
            scoreType=cv2.ORB_HARRIS_SCORE)

        imgTpt = cv2.imread(self.PATH_TEMPLATE)
        w, h, brd = *self.SHAPE_OUT, self.PAD_BORDER
        matOrbTpt = cv2.getAffineTransform(
            np.float32([[0, 0], [imgTpt.shape[0], 0], [0, imgTpt.shape[1]]]),
            np.float32([[brd, brd], [w-brd, brd], [brd, h-brd]]))
        self.imgTpt = cv2.warpAffine(imgTpt, matOrbTpt, [w, h],
                                     borderValue=self.COLOR_BACKGROUND)
        self.features_tpl = self.detect_features(self.imgTpt, orbL)

        self.flann = cv2.FlannBasedMatcher(
            self.index_params,
            self.search_params)

    def draw(self, img, features=None):
        if features == None:
            features = self.detect_features(img)
        return cv2.drawKeypoints(img, features['kps'], 0,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def get_matrix(self, img, plot=False):
        features = self.detect_features(img)
        if plot:
            self.imgDraw = img.copy()
            self.imgDraw = self.draw(self.imgDraw, features)

        assert len(features['kps']) > 0, "No features found"
        pts0, pts1, matches = self.match_features(features, self.features_tpl)

        if plot:
            self.imgDraw = cv2.drawMatches(img, features['kps'],
                                           self.imgTpt, self.features_tpl['kps'], matches, None)
        assert len(pts0) > self.MIN_MATCHES, "Not enough matches"

        return cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)[0]

    def detect(self, img, plot=False):
        return cv2.warpPerspective(
            img, self.get_matrix(img, plot), self.SHAPE_OUT)

    def match_features(self, features0, features1):
        LOWES_RATIO = 0.7
        matches = []  # good matches as per Lowe's ratio test
        all_matches = self.flann.knnMatch(
            features0['des'], features1['des'], k=2)

        for m, n in all_matches:
            if m.distance < LOWES_RATIO * n.distance:
                matches.append(m)

        matched_pts0, matched_pts1 = [], []
        for m in matches:
            matched_pts0.append(features0['kps'][m.queryIdx].pt)
            matched_pts1.append(features1['kps'][m.trainIdx].pt)

        return np.float32(matched_pts0).reshape(-1, 1, 2), \
            np.float32(matched_pts1).reshape(-1, 1, 2), matches

    def detect_features(self, img, orb=None):
        if orb is None:
            orb = self.orb
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = orb.detectAndCompute(gray_img, None)
        return dict(kps=kps, des=des)


class SpotObject:
    def __init__(self, rect, shape, tokenSize):
        self.rect = rect
        self.shape = np.array(shape)
        self.tokenSize = tokenSize

    def __add__(self, other):  # translation
        return SpotObject(self.rect + np.r_[other, 0, 0], self.shape, self.tokenSize)

    def __mul__(self, other):  # scale
        return SpotObject(self.rect * np.r_[other, other, other, other], self.shape,
                          self.tokenSize * np.r_[other, other])

    def draw(self, img):
        pt0 = self.rect[:2]
        shapeSafe = np.array([s if s > 1 else 2 for s in self.shape])
        sz = self.rect[2:] / (shapeSafe[::-1] - 1)
        corners = np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.tokenSize / 2
        for iy in range(self.shape[0]):
            for ix in range(self.shape[1]):
                # pt = tuple(np.int32(pt0 + sz * [ix, iy]))
                # cv2.circle(img, pt, 10, (0, 0, 255), -1)
                pts = np.int32(pt0 + sz * [ix, iy] + corners).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], True, color=(
                    255, 255, 255), thickness=1)

    def crop(self, img, idx):
        rect = self.get_rect(idx)
        # pt0 = self.rect[:2]
        # shapeSafe = np.array([s if s > 1 else 2 for s in self.shape])
        # sz = self.rect[2:] / (shapeSafe[::-1] - 1)
        # iy, ix = idx
        # pt0 = tuple(
        #     np.int32(pt0 + sz * [ix, iy] - self.tokenSize / 2))
        # pt1 = tuple(np.int32(pt0 + self.tokenSize))
        return img[rect[1]:rect[3], rect[0]:rect[2]]

    def get_rect(self, idx):
        pt0 = self.rect[:2]
        shapeSafe = np.array([s if s > 1 else 2 for s in self.shape])
        sz = self.rect[2:] / (shapeSafe[::-1] - 1)
        iy, ix = idx
        pt0 = tuple(
            np.int32(pt0 + sz * [ix, iy] - self.tokenSize / 2))
        pt1 = tuple(np.int32(pt0 + self.tokenSize))
        return pt0 + pt1

    def get_features(self, im):
        # assume BGR image
        hist = cv2.calcHist([im], [0, 1, 2], None, [
                            4, 4, 4], [0, 256, 0, 256, 0, 256])
        return hist.reshape(-1) / np.sum(hist)

    def features_from_image(self, img):
        feat = []
        for iy in range(self.shape[0]):
            for ix in range(self.shape[1]):
                im = self.crop(img, (iy, ix))
                feat.append(self.get_features(im))
        return feat

    def get_diff_grid(self, imgDiff):
        val = np.zeros((self.shape[0], self.shape[1]))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                val[i, j] = self.crop(imgDiff, (i, j)).mean()
        return val


class TileClassifier:
    IMAGE_SIZE = 34
    DEVICE = 'cpu'
    class_names = ['background', 'black', 'blue',
                   'first', 'red', 'white', 'yellow']

    def __init__(self):
        # Load the trained model.
        self.model = build_model(pretrained=False, fine_tune=False,
                                 num_classes=len(self.class_names))
        checkpoint = torch.load(
            MODEL_PATH, map_location=self.DEVICE)
        print('Loading trained model weights...')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(self.DEVICE)

        # Forward pass throught the image.
        outputs = self.model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, classes = torch.max(probs, 1)
        outputs = outputs.detach().numpy()
        names = self.class_names[classes[0]]
        
        return classes.detach().numpy()[0], conf.detach().numpy()[0], names

    def predict_batch(self, image):
        imgs = []
        for im in image:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.transform(im)
            imgs.append(im)
        image = torch.stack(imgs)
        image = image.to(self.DEVICE)

        # Forward pass throught the image.
        outputs = self.model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, classes = torch.max(probs, 1)
        # outputs = outputs.detach().numpy()
        names = [self.class_names[y] for y in classes]
        return classes.detach().numpy(), conf.detach().numpy(), names


class BoardDetector:
    SHAPE_OUT = (500, 500)
    PAD_BORDER = 30
    MAX_REPROJ_ERROR = 50
    MAX_MSE_ERROR = 5_000
    PATH_TEMPLATE = PATH_TEMPLATE
    COLOR_BACKGROUND = (185, 197, 203)

    def __init__(self):
        self.detORB = BoardDetector_ORB(
            SHAPE_OUT=self.SHAPE_OUT, PATCH_SIZE=31)
        self.detYOLO = BoardDetector_YOLO(
            SHAPE_OUT=self.SHAPE_OUT, PAD_BORDER=50)

        imgTpt = cv2.imread(self.PATH_TEMPLATE)
        w, h, brd = *self.SHAPE_OUT, self.PAD_BORDER
        mat = cv2.getAffineTransform(
            np.float32([[0, 0], [imgTpt.shape[0], 0], [0, imgTpt.shape[1]]]),
            np.float32([[brd, brd], [w-brd, brd], [brd, h-brd]]))
        self.imgTpt = cv2.warpAffine(imgTpt, mat, [w, h],
                                     borderValue=self.COLOR_BACKGROUND)

        scale = self.SHAPE_OUT[0] - 2 * self.PAD_BORDER
        trans = [self.PAD_BORDER] * 2
        self.spotScore = SpotObject(
            [0.0656, 0.0276, 0.8656, 0.278], (6, 20), (0.0412, 0.052)) * scale + trans
        self.spotLines = SpotObject(
            [0.078, 0.3956, 0.3544, 0.3588], (5, 5), (0.0792, 0.0792)) * scale + trans
        self.spotGrid = SpotObject(
            [0.5624, 0.3956, 0.3512, 0.3588], (5, 5), (0.0792, 0.0792)) * scale + trans
        self.spotFloor = SpotObject(
            [0.078, 0.9032, 0.5756, 0.0792], (1, 7), (0.0792, 0.0792)) * scale + trans

        self.tileClassifier = TileClassifier()

    def estimate_K(self, img):
        h, w, c = img.shape
        f = 0.6 * w
        return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], np.float32)

    def detect_tiles(self, img, plot=False):
        imgs, ups = self.detect(img)
        imt = []
        for im, up in zip(imgs, ups):
            imt += [(self.spotGrid + up).crop(im, (i//5, i % 5)) for i in range(25)] + \
                [(self.spotLines + up).crop(im, (i//5, i % 5)) for i in range(25) if i % 5 >= 4 - i//5] + \
                [(self.spotFloor + up).crop(im, (0, i)) for i in range(7)]

        # more efficient than single predictions
        cls, conf, clsName = self.tileClassifier.predict_batch(imt)
        numPlayers = len(imgs)
        tileDict = {'background': -1, 'black': 3, 'blue': 0,
                    'first': 5, 'red': 2, 'white': 4, 'yellow': 1}
        cls = np.reshape([tileDict[c] for c in clsName], [numPlayers, -1])
        player = []
        # tptGray = cv2.cvtColor(
        #     self.imgTpt, cv2.COLOR_BGR2GRAY).astype(np.float32)
        tptGray = cv2.cvtColor(
            self.imgTpt, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
        vals = []
        for i in range(numPlayers):
            # im = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            im = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV)[
                :, :, 2].astype(np.float32)
            imgdiff = cv2.absdiff(tptGray, im)
            val = self.spotScore.get_diff_grid(imgdiff) + \
                (self.spotScore + ups[i]).get_diff_grid(imgdiff)
            vals.append(val)
            idx = val.argmax()
            iscore, jscore = idx // self.spotScore.shape[1], idx % self.spotScore.shape[1]
            if iscore > 0:
                score = jscore + 1 + (iscore - 1) * self.spotScore.shape[1]
            else:
                score = 0

            lines = cls[i, 25:-7].tolist()
            lineNum, lineColor = [], []
            for row in range(5):
                lines, tiles = lines[row+1:], lines[:row+1]
                lineColor.append(tiles[-1])
                lineNum.append(tiles.count(
                    tiles[-1]) if tiles[-1] != -1 else 0)

            player.append(dict(
                score=score,
                grid=cls[i, :25].reshape(5, 5),
                lineColor=lineColor,
                lineNum=lineNum,
                floor=cls[i, -7:]
            ))

        if plot:
            for im, p in zip(imgs, player):
                self.mark_board(im, p)

        return player, imgs, vals

    def mark_board(self, im, player):
        NUM_ROWS = 5
        classLabelDict = {-1: '', 3: 'K', 0: 'B',
                          5: 'F', 2: 'R', 4: 'W', 1: 'Y'}
        classColorDict = {-1: (255, 255, 255), 3: (200, 200, 200), 0: (255, 150, 150),
                          5: (255, 255, 255), 2: (100, 100, 255), 4: (255, 255, 255),
                          1: (100, 255, 255)}

        def label_tile(spot, row, col, color):
            rect = np.int32(spot.get_rect((row, col)))
            cv2.rectangle(im, rect[:2], rect[-2:], (200, 200, 200), 1)
            cv2.putText(im, classLabelDict[color],
                        rect[:2] + [5+1, 20+1], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)
            cv2.putText(im, classLabelDict[color],
                        rect[:2] + [5-1, 20-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        classColorDict[color], 2)

        for row in range(NUM_ROWS):
            for col in range(NUM_ROWS):
                label_tile(self.spotGrid, row, col, player['grid'][row, col])

                if row >= NUM_ROWS - 1 - col:
                    color = player['lineColor'][row] if NUM_ROWS - \
                        col-1 < player['lineNum'][row] else -1
                    label_tile(self.spotLines, row, col, color)

        for col in range(7):
            label_tile(self.spotFloor, 0, col, player['floor'][col])

        if player['score'] > 0:
            i, j = player['score'] // 20 + 1, (player['score'] - 1) % 20
        else:
            i, j = 0, 0
        rect = np.int32(self.spotScore.get_rect((i, j)))
        cv2.rectangle(im, rect[:2], rect[-2:], (200, 200, 200), 1)
        cv2.putText(im, f"{player['score']}",
                    rect[:2] + [3+1, 10+1], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,0), 2)
        cv2.putText(im, f"{player['score']}",
                    rect[:2] + [3-1, 10-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

    def detect(self, img, plot=False):
        if plot:
            self.imgDraw = img.copy()
        brd, brdOrb = self.PAD_BORDER, self.detORB.PAD_BORDER
        h, w = self.SHAPE_OUT
        dst = np.float32(
            [[brdOrb, h-brdOrb], [w-brdOrb, h-brdOrb], [w-brdOrb, brdOrb], [brdOrb, brdOrb]])
        # board corners in iboard coord
        src = np.float32(
            [[brd, h-brd], [w-brd, h-brd], [w-brd, brd], [brd, brd]])
        matrixBorder = cv2.getPerspectiveTransform(dst, src)

        # board corners in norm board coord
        objpts = np.array(
            [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]], np.float32)
        objCoords = np.float32(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 2
        objUp = np.array([[0.5, 0.5, 0], [0.5, 0.5, -0.033]], np.float32)
        K = self.estimate_K(img)

        imgs = []
        upVector = []
        brd = self.PAD_BORDER
        for matrixYOLO in self.detYOLO.get_matrix(img, plot):
            image = cv2.warpPerspective(
                img, matrixYOLO, self.detYOLO.SHAPE_OUT, borderMode=cv2.BORDER_REPLICATE)
            try:
                matrixORB = self.detORB.get_matrix(image, plot)
            except AssertionError as e:
                if e == "Not enough matches" or e == "No features found":
                    warn("ORB did not find matches")
                    continue
                # raise e

            matRawToIBoard = matrixBorder @ matrixORB @ matrixYOLO
            im = cv2.warpPerspective(
                img, matRawToIBoard, self.SHAPE_OUT)

            if mse(self.imgTpt, im) > self.MAX_MSE_ERROR:
                print(f"MSE too high: {mse(self.imgTpt, im)}")
                continue

            # estimate upVector
            # from iboard coordinates to raw
            corners = cv2.perspectiveTransform(
                src[:, np.newaxis, :], np.linalg.pinv(matRawToIBoard))

            # TODO use method that works with non-square boards
            retval, rvec, tvec = cv2.solvePnP(
                objpts, np.float32(corners.copy()), K, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)

            cornersReproj = cv2.projectPoints(objpts, rvec, tvec, K, None)[0]
            err = np.mean(np.sum((cornersReproj - corners)**2, 2)**0.5)
            if err > self.MAX_REPROJ_ERROR:
                print(f"Reprojection error too high: {err}")
                continue

            ptsUp, _ = cv2.projectPoints(objUp, rvec, tvec, K, None)
            ptsUp = cv2.perspectiveTransform(
                ptsUp, matRawToIBoard)[:, 0, :]

            imgs.append(im)
            upVector.append(ptsUp[1] - ptsUp[0])

            if plot:
                brdCoords, _ = cv2.projectPoints(
                    objCoords, rvec, tvec, K, None)
                cv2.polylines(self.imgDraw, [np.int32(brdCoords[[0, 1]])],
                              True, color=(0, 0, 255), thickness=15)
                cv2.polylines(self.imgDraw, [np.int32(brdCoords[[0, 2]])],
                              True, color=(0, 255, 0), thickness=15)
                cv2.polylines(self.imgDraw, [np.int32(brdCoords[[0, 3]])],
                              True, color=(255, 0, 0), thickness=15)
                for c in corners:
                    self.imgDraw = cv2.circle(self.imgDraw, tuple(
                        c[0].astype(int)), 30, (0, 0, 255), -1)

        return imgs, upVector


def feature_matching(features0, features1):
    matches = []  # good matches as per Lowe's ratio test
    if (features0.des is not None and len(features0.des) > 2):
        all_matches = flann.knnMatch(
            features0.des, features1.des, k=2)
        try:
            for m, n in all_matches:
                if m.distance < LOWES_RATIO * n.distance:
                    matches.append(m)
        except ValueError:
            pass
        if (len(matches) > MIN_MATCHES):
            features0.matched_pts = np.float32(
                [features0.kps[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            features1.matched_pts = np.float32(
                [features1.kps[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
    return matches


def corners_from_dm(frame, border):
    """Find corners of board using descriptor (ORB) matching"""
    h, w, c = frame.shape
    brd = border

    features1 = FeatureExtraction(frame)
    matches = feature_matching(_features0, features1)

    if len(features1.matched_pts) == 0:
        return None

    return cv2.findHomography(
        features1.matched_pts, _features0.matched_pts, cv2.RANSAC, 5.0)[0]


def mse(img1, img2):
    """Similarity betweek 2 board images"""
    h, w, c = img1.shape
    diff = cv2.subtract(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32),
                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32))
    err = np.sum(diff**2)
    # calculate nmse
    # nmse = err/(float(h*w)) / 255**2
    mse = err/(float(h*w))
    return mse


def get_boards(img, plot=False):
    """
        get_boards: 

        Input:
            - img: BGR image of board
            - plot: whether to plot intermediate results
        Output:
            - imgBoard: list of N BGR images of each board
            - maskMerge: mask of all boards
            - matrices: list of N matrices that transform each board to C1
    """
    imgClean = img.copy()
    masks, boxes = segment(img, plot)
    maskMerge = np.sum(np.uint8(masks), axis=0).astype(np.uint8)

    h1, w1 = masks[0].shape
    hf, wf, _ = img.shape
    BRD, FSZ = 80, 800
    dst = np.float32([[BRD, BRD], [FSZ-BRD, BRD],
                     [FSZ-BRD, FSZ-BRD], [BRD, FSZ-BRD]])

    BRD2, FSZ2 = 0, 640  # output board resolution
    sz = FSZ2 / 640
    # C1 to C0
    matrix3 = cv2.getPerspectiveTransform(dst, np.float32([
        [0, 0], [FSZ2, 0], [FSZ2, FSZ2], [0, FSZ2]]))
    BORDER_COLOR = (81, 56, 40)

    imgBoard, matrices = [], []
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        src = polygon_from_mask(np.uint8(mask))
        corners = src * [wf/w1, hf/h1]  # in raw coordinates
        # raw to C1'
        matrix = cv2.getPerspectiveTransform(
            np.float32(corners), dst)
        frame = cv2.warpPerspective(
            imgClean, matrix, (FSZ, FSZ), borderMode=cv2.BORDER_REPLICATE)

        # C1' to C1
        matrix2 = corners_from_dm(frame, BRD)
        if matrix2 is None:
            warn('No ORB match found')
            continue
        frame = cv2.warpPerspective(
            imgClean, matrix3 @ matrix2 @ matrix, (FSZ2, FSZ2), borderValue=BORDER_COLOR)

        c0 = corners.mean(0).astype(int)
        wid = np.mean(np.sum(np.diff(corners, axis=0)**2, 1)**0.5).astype(int)
        # print(c0, wid)

        # upVector = np.linalg.solve(matrix3 @ matrix2 @ matrix, [c0[0], c0[1]-wid//30, 1]) - \
        #     np.linalg.solve(matrix3 @ matrix2 @ matrix, [c0[0], c0[1], 1])

        # upVector = matrix3 @ matrix2 @ matrix @ [c0[0], c0[1]-wid/30, 1] - \
        #     matrix3 @ matrix2 @ matrix @ [c0[0], c0[1], 1]
        objpts = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
        objUp = np.array([[0.5, 0.5, 0], [0.5, 0.5, -0.03]], np.float32)
        imgpts = np.float32(corners)
        h, w, c = img.shape
        f = 0.6 * w
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], np.float32)
        retval, rvec, tvec = cv2.solvePnP(
            objpts, imgpts, K, None,
            np.array([-0.8, 0, 0], np.float32),
            np.array([0, 0, 5], np.float32), True)
        ptsUp, _ = cv2.projectPoints(objUp, rvec, tvec, K, None)
        ptsUp = cv2.perspectiveTransform(
            ptsUp, matrix3 @ matrix2 @ matrix)[:, 0, :]
        upVector = ptsUp[1] - ptsUp[0]

        valMatch = mse(frame, _template0)
        if plot:
            cv2.polylines(maskMerge, [np.int32(src)],
                          True, color=2, thickness=int(np.ceil(1*sz)))
            for pt in src:
                cv2.circle(maskMerge, np.int32(pt), int(
                    3*sz), color=2, thickness=-1)
            cv2.putText(frame, f"{int(valMatch)}", (int(10*sz), int(40*sz)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2*sz, (50, 50, 255), int(3*sz))
            cv2.arrowedLine(frame, ptsUp[0].astype(int),
                            ptsUp[1].astype(int), (255, 0, 0), 7)
            cv2.line(img, [c0[0], c0[1]-wid//30], [c0[0], c0[1]],
                     color=(255, 0, 0), thickness=11)

            imgptsRepr, _ = cv2.projectPoints(objpts, rvec, tvec, K, None)
            objUpLong = np.array([[0.5, 0.5, 0], [0.5, 0.5, -0.5]], np.float32)
            ptsUpLong, _ = cv2.projectPoints(objUpLong, rvec, tvec, K, None)

            cv2.polylines(img, [np.int32(imgptsRepr)], True, (128, 0, 128), 20)
            cv2.circle(img, tuple(imgpts[0].astype(int)), 30, (0, 0, 255), -1)
            cv2.circle(img, tuple(imgpts[1].astype(int)), 50, (0, 0, 255), -1)
            cv2.arrowedLine(img, tuple(*ptsUpLong[0].astype(int)),
                            tuple(*ptsUpLong[1].astype(int)), (255, 0, 0), 30)
            cv2.putText(img, f"{i}",
                        tuple(imgpts[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 20)

        imgBoard.append(frame)
        matrices.append(upVector)
        # matrices.append(corners)

    return imgBoard, maskMerge, matrices


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

    return results[0].masks.data, results[0].boxes.data


def signed_area(u, v):
    return u[0] * v[1] - u[1] * v[0]


# TODO test with polygons besides N=4
def polygon_from_mask(mask, numCorners=4):
    # [s0, v] * N
    # s0: vertex
    # v: direction vector of edge, oriented toward +z
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get largest blob
    extent = [xy[:, 0, 0].max() - xy[:, 0, 0].max() +
              xy[:, 0, 1].max() - xy[:, 0, 1].min() for xy in contours]
    xy = contours[np.argmax(extent)][:, 0, :]

    # resample points to make consecutive points equally distant
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
                                   labels, criteria, 15, flags)

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
    TPT_SHAPE = (w - 2*border, h - 2*border)
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
