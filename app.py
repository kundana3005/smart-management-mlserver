from flask import Flask, jsonify
import cv2
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "🚀 ML Server is running successfully!"})

@app.route('/compare')
def compare_images():
    try:
        before = cv2.imread("dataset/before.png")
        after = cv2.imread("dataset/after.png")

        if before is None or after is None:
            return jsonify({"error": "Missing image files in dataset folder"}), 400

        # Resize both images
        before = cv2.resize(before, (600, 400))
        after = cv2.resize(after, (600, 400))

        # Convert to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, diff = ssim(before_gray, after_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Invert and threshold
        diff_inv = cv2.bitwise_not(diff)
        _, thresh = cv2.threshold(diff_inv, 30, 255, cv2.THRESH_BINARY)

        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = diff_inv.size
        change_percentage = (changed_pixels / total_pixels) * 100

        return jsonify({
            "cleanup_progress": round(change_percentage, 2),
            "message": "✅ Comparison complete!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
