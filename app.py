import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import io

class PersonCounterDL:
    """
    Person detection and counting system with face validation
    """
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades
        
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_eye.xml')
        )
        
        self.use_eye_detection = not self.eye_cascade.empty()
    
    def _is_valid_face(self, image, gray, x, y, w, h):
        """Validate if detection is a real face"""
        aspect_ratio = float(w) / h if h > 0 else 0
        if not (0.6 < aspect_ratio < 1.4):
            return False
        
        roi_bgr = image[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask1 = cv2.inRange(roi_hsv, lower_skin, upper_skin)
        
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        skin_mask2 = cv2.inRange(roi_hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        skin_percentage = np.count_nonzero(skin_mask) / (w * h)
        
        if skin_percentage < 0.15:
            return False
        
        roi_gray = gray[y:y+h, x:x+w]
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_percentage = np.count_nonzero(edges) / (w * h)
        
        if edge_percentage < 0.05:
            return False
        
        contrast = roi_gray.std()
        if contrast < 15:
            return False
        
        return True
    
    def _detect_faces_haar(self, image):
        """Detect faces with validation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(300, 300)
        )
        
        validated_faces = []
        for (x, y, w, h) in faces:
            if self._is_valid_face(image, gray, x, y, w, h):
                validated_faces.append((x, y, w, h, 0.9))
        
        return validated_faces
    
    def _nms(self, boxes, overlap_thresh=0.3):
        """Non-maximum suppression"""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []
        
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            remaining = []
            for box in boxes:
                x1, y1, w1, h1, conf1 = current
                x2, y2, w2, h2, conf2 = box
                
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    union = w1 * h1 + w2 * h2 - inter
                    iou = inter / union if union > 0 else 0
                    
                    if iou <= overlap_thresh:
                        remaining.append(box)
                else:
                    remaining.append(box)
            
            boxes = remaining
        
        return keep
    
    def count_persons(self, image, visualize=True):
        """Main detection function"""
        h, w = image.shape[:2]
        if w > 1280:
            scale = 1280 / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        haar_faces = self._detect_faces_haar(image)
        all_detections = haar_faces
        
        final_detections = self._nms(all_detections, overlap_thresh=0.3)
        final_detections = sorted(final_detections, key=lambda x: x[0])
        
        count = len(final_detections)
        annotated = image.copy() if visualize else None
        
        if visualize:
            for idx, (x, y, bw, bh, conf) in enumerate(final_detections, 1):
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cx, cy = x + bw // 2, y + bh // 2
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                label = f"P{idx}"
                cv2.putText(annotated, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.putText(annotated, f"Total: {count} person(s)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return count, annotated, final_detections


# ============= STREAMLIT APP =============

st.set_page_config(page_title="Person Counter", layout="wide", initial_sidebar_state="expanded")

st.title("👥 Person Counter - Computer Vision")
st.markdown("_Detect and count people in images using OpenCV & Haar Cascades_")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_info = st.info("ℹ️ This detector uses Haar Cascades with multi-level face validation")

# Initialize detector
@st.cache_resource
def load_detector():
    return PersonCounterDL()

detector = load_detector()

# Main content
st.subheader("📤 Upload Your Image")
    
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process
    with st.spinner("🔍 Analyzing image..."):
        count, annotated, detections = detector.count_persons(img_array, visualize=True)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Detection Results")
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Persons", count)
    with col2:
        st.metric("Confidence", "0.90")
    with col3:
        st.metric("Method", "Haar Cascade")
    
    # Detailed results
    if detections:
        st.subheader("📋 Detailed Detections")
        
        detail_cols = st.columns(4)
        headers = ["Person", "Position (x,y)", "Size (w×h)", "Center"]
        
        for i, header in enumerate(headers):
            with detail_cols[i % 4]:
                st.write(f"**{header}**")
        
        for idx, (x, y, w, h, conf) in enumerate(detections, 1):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"#{idx}")
            with col2:
                st.write(f"({x}, {y})")
            with col3:
                st.write(f"{w}×{h}")
            with col4:
                st.write(f"({x + w//2}, {y + h//2})")
    
    # Download button
    result_image = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Result",
        data=buf.getvalue(),
        file_name=f"person_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 <b>Person Counter</b> | Computer Vision Project</p>
    <p>Built with OpenCV, Python, and Streamlit</p>
</div>
""", unsafe_allow_html=True)