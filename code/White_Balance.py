import cv2
import numpy as np
import matplotlib.pyplot as plt

# 灰度世界法
def gray_world_white_balance(image):
    rgb_image = np.float32(image)
    
    r, g, b = cv2.split(rgb_image)
    
    Ravg = np.mean(r)
    Gavg = np.mean(g)
    Bavg = np.mean(b)
    
    M = max(Ravg, Gavg, Bavg)
    
    Kr = M / Ravg
    Kg = M / Gavg
    Kb = M / Bavg
    
    r = np.clip(r * Kr, 0, 255).astype(np.uint8)
    g = np.clip(g * Kg, 0, 255).astype(np.uint8)
    b = np.clip(b * Kb, 0, 255).astype(np.uint8)
    
    new_image = cv2.merge([r, g, b])
    
    return new_image


# 完美反射法
def perfect_reflection_white_balance(image):
    image = np.float32(image)

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    sum_RGB = R + G + B

    sum_RGB_int = np.int32(sum_RGB)
    
    histogram = np.bincount(sum_RGB_int.ravel(), minlength=256*3)
    cumulative_sum = np.cumsum(histogram[::-1])[::-1]
    total_pixels = image.size / 3
    threshold_index = np.where(cumulative_sum > 0.30 * total_pixels)[0][0]
    threshold = threshold_index

    bright_pixels = sum_RGB > threshold
    R_avg = np.mean(R[bright_pixels])
    G_avg = np.mean(G[bright_pixels])
    B_avg = np.mean(B[bright_pixels])

    M = max(R_avg, G_avg, B_avg)
    Kr, Kg, Kb = M / R_avg, M / G_avg, M / B_avg

    R_new = R * Kr
    G_new = G * Kg
    B_new = B * Kb

    R_new = np.clip(R_new, 0, 255)
    G_new = np.clip(G_new, 0, 255)
    B_new = np.clip(B_new, 0, 255)

    balanced_image = np.dstack([R_new, G_new, B_new])
    balanced_image = np.uint8(balanced_image)

    return balanced_image