import cv2
from skimage.metrics import structural_similarity as ssim

before = cv2.imread("dataset/before.png")
after = cv2.imread("dataset/after.png")

before = cv2.resize(before, (600, 400))
after = cv2.resize(after, (600, 400))

before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

score, diff = ssim(before_gray, after_gray, full=True)
diff = (diff * 255).astype("uint8")

diff_inv = cv2.bitwise_not(diff)
_, thresh = cv2.threshold(diff_inv, 30, 255, cv2.THRESH_BINARY)

changed_pixels = cv2.countNonZero(thresh)
total_pixels = diff_inv.size
change_percentage = (changed_pixels / total_pixels) * 100

print(f"🧹 Cleanup Progress: {round(change_percentage, 2)}% area changed")