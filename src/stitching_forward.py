import cv2
import numpy as np
import os
from pathlib import Path

# 1. 전역 변수 및 마우스 콜백 (이전과 동일)
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
    cv2.setMouseCallback("Select Points", mouse_callback)
    print(f"\n>>> {title}: 점을 찍고 'q'를 누르세요.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32) / scale

# 2. Forward Warping 핵심 로직
def forward_warping(img_src, H, canvas_shape, offset_x):
    h_src, w_src = img_src.shape[:2]
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    
    # 원본 이미지의 모든 픽셀을 순회 (x, y) -> (x', y')
    for y in range(h_src):
        for x in range(w_src):
            # 동차 좌표계 변환
            p = np.array([x, y, 1.0])
            # p' = H * p
            p_prime = np.dot(H, p)
            
            # Cartesian 좌표 복원 (w로 나누기)
            if p_prime[2] != 0:
                x_prime = int(p_prime[0] / p_prime[2]) + offset_x
                y_prime = int(p_prime[1] / p_prime[2])
                
                # 캔버스 범위 체크 후 색상 대입 
                if 0 <= x_prime < canvas.shape[1] and 0 <= y_prime < canvas.shape[0]:
                    canvas[y_prime, x_prime] = img_src[y, x]
    return canvas

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
        h, w = i_c.shape[:2]
        canvas_size = (h, w * 3, 3)

        # 대응점 구하기
        pts_l = get_manual_points(i_l, "Left", view_scale)
        pts_cl = get_manual_points(i_c, "Center(L)", view_scale)
        pts_r = get_manual_points(i_r, "Right", view_scale)
        pts_cr = get_manual_points(i_c, "Center(R)", view_scale)

        # Homography 계산
        H_lc, _ = cv2.findHomography(pts_l, pts_cl)
        H_rc, _ = cv2.findHomography(pts_r, pts_cr)

        # Forward Warping 수행
        print("\nForward Warping 진행 중... (시간이 다소 소요될 수 있습니다)")
        warped_l = forward_warping(i_l, H_lc, canvas_size, w)
        warped_r = forward_warping(i_r, H_rc, canvas_size, w)

        # 합성
        result = warped_l.copy()
        mask_r = (warped_r > 0).any(axis=2)
        result[mask_r] = warped_r[mask_r]
        result[0:h, w:w*2] = i_c

        cv2.imwrite(str(output_dir / "forward_result.jpg"), result)
        print("\n✅ Forward Warping 결과 저장 완료!")
        cv2.imshow("Forward Result", cv2.resize(result, None, fx=0.3, fy=0.3))
        cv2.waitKey(0)

    except Exception as e:
        raise RuntimeError(f"영상 정합 실행 실패: {e}") from e
