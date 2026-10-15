"""Coursework: manual DLT/SVD and inverse warping with bilinear sampling."""
import numpy as np

def compute_homography(pts1, pts2):
    """
    DLT(Direct Linear Transformation)를 이용한 호모그래피 행렬 계산
    pts1, pts2: (N, 2) 형태의 대응점 좌표 (N >= 4)
    """
    A = []
    for i in range(len(pts1)):
        x, y = pts1[i]
        u, v = pts2[i]
        # 강의자료 Lec 10, Page 22의 Ah = 0 행렬 구성
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    
    # SVD를 이용해 Ah = 0의 해(가장 작은 특이값에 대응하는 오른쪽 특이벡터)를 구함
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :]
    H = L.reshape(3, 3)
    
    # H[2, 2]를 1로 정규화
    return H / H[2, 2]

def bilinear_interpolation(img, x, y):
    """
    Bilinear Interpolation을 이용한 픽셀 값 계산 (Lec 10, Page 12)
    """
    h, w, c = img.shape
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    
    # 이미지 범위를 벗어나는 경우 처리
    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return np.zeros(c)
    
    a = x - x0
    b = y - y0
    
    # 수식: (1-a)(1-b)f(i,j) + a(1-b)f(i+1,j) + (1-a)bf(i,j+1) + abf(i+1,j+1)
    val = (1-a)*(1-b)*img[y0, x0] + a*(1-b)*img[y0, x1] + \
          (1-a)*b*img[y1, x0] + a*b*img[y1, x1]
    
    return val

def warp_image(img, H, output_shape):
    """
    Backward Warping을 이용한 영상 변환 (Lec 10, Page 8)
    """
    h_out, w_out = output_shape
    result = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H) # Backward mapping을 위해 역행렬 사용
    
    for v_out in range(h_out):
        for u_out in range(w_out):
            # 결과 이미지 좌표 -> 원본 이미지 좌표 역산
            dst_pt = np.array([u_out, v_out, 1.0])
            src_pt = H_inv @ dst_pt
            src_pt /= src_pt[2] # Homogeneous coordinate 정규화
            
            u_src, v_src = src_pt[0], src_pt[1]
            
            # Interpolation을 통해 픽셀 값 결정
            result[v_out, u_out] = bilinear_interpolation(img, u_src, v_src)
            
    return result
