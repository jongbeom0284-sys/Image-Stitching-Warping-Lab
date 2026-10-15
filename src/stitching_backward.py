import cv2
import numpy as np
import os
from pathlib import Path

# 전역 변수
points = []
display_img = None
clean_img = None

def redraw_points():
    global display_img, points, clean_img
    display_img = clean_img.copy()
    for i, (px, py) in enumerate(points):
        cv2.circle(display_img, (int(px), int(py)), 5, (0, 0, 255), -1)
        cv2.putText(display_img, str(i + 1), (int(px) + 10, int(py) + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Select Points", display_img)

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        redraw_points()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop()
            redraw_points()

def get_manual_points(img, title, scale):
    global display_img, points, clean_img
    points = []
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    clean_img = resized.copy()
    display_img = resized.copy()
    
    cv2.namedWindow("Select Points", cv2.WINDOW_NORMAL) 
    cv2.imshow("Select Points", display_img)
    cv2.setWindowTitle("Select Points", title)
    cv2.setMouseCallback("Select Points", mouse_callback)
    
    print(f"\n>>> {title}: 대응점을 클릭하고 'q'를 누르세요. (우클릭: 취소)")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32) / scale

if __name__ == "__main__":
    try:
        path = Path(os.environ.get("STITCHING_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
        output_dir = Path(__file__).resolve().parents[1] / "outputs"
        output_dir.mkdir(exist_ok=True)
        i_l = cv2.imread(os.path.join(path, 'left.jpg'))
        i_c = cv2.imread(os.path.join(path, 'center.jpg'))
        i_r = cv2.imread(os.path.join(path, 'right.jpg'))

        if any(img is None for img in (i_l, i_c, i_r)):
            raise FileNotFoundError(f"Expected left.jpg, center.jpg, right.jpg in {path}")

        view_scale = 0.2 

        # 1. 대응점 찾기 (각 이미지를 센터 기준으로 매칭)
        pts_l = get_manual_points(i_l, "1. Left Image", view_scale)
        pts_cl = get_manual_points(i_c, "2. Center Image (for Left)", view_scale)
        pts_r = get_manual_points(i_r, "3. Right Image", view_scale)
        pts_cr = get_manual_points(i_c, "4. Center Image (for Right)", view_scale)

        # 2. Homography 계산 (DLT 방식)
        H_lc, _ = cv2.findHomography(pts_l, pts_cl)
        H_rc, _ = cv2.findHomography(pts_r, pts_cr)

        # H_lc 계산 후 출력
        H_lc, _ = cv2.findHomography(pts_l, pts_cl)
        print("--- Homography Matrix (Left to Center) ---")
        print(H_lc)

        # H_rc 계산 후 출력
        H_rc, _ = cv2.findHomography(pts_r, pts_cr)
        print("--- Homography Matrix (Right to Center) ---")
        print(H_rc)

        # 3. Backward Warping 수행 
        # Center 이미지가 가운데 오도록 가로 3배 크기의 캔버스 설정
        h, w = i_c.shape[:2]
        # 가로 오프셋 행렬 (중앙 이미지를 캔버스의 가운데로 밀어줌)
        T = np.array([[1, 0, w], [0, 1, 0], [0, 0, 1]]) 

        # OpenCV의 warpPerspective는 기본적으로 Backward Warping을 수행함 
        # INTER_LINEAR는 Bilinear Interpolation을 의미함
        warped_l = cv2.warpPerspective(i_l, T @ H_lc, (w * 3, h), flags=cv2.INTER_LINEAR)
        warped_r = cv2.warpPerspective(i_r, T @ H_rc, (w * 3, h), flags=cv2.INTER_LINEAR)
    
        # 4. 합성 및 결과 생성
        result = warped_l.copy()
        # 픽셀 값이 0이 아닌 부분을 덮어씌움 (Blending 기초)
        mask_r = (warped_r > 0).any(axis=2)
        result[mask_r] = warped_r[mask_r]
    
        # 중앙 이미지는 변환 없이 지정된 위치에 배치
        result[0:h, w:w*2] = i_c

        # 최종 결과 저장
        cv2.imwrite(str(output_dir / "backward_warping_result.jpg"), result)
        print("\n✅ Backward Warping 기반 스티칭 완료!")
        cv2.imshow("Result", cv2.resize(result, None, fx=0.3, fy=0.3))
        cv2.waitKey(0)

    except Exception as e:
        raise RuntimeError(f"영상 정합 실행 실패: {e}") from e
