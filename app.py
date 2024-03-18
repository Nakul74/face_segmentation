import streamlit as st
import detect_face
import face_seg

def save_image(uploaded_image, file_name):
    file_ext = file_name.split('.')[-1]
    file_name = f'input_img.{file_ext}'
    with open(file_name, "wb") as f:
        f.write(uploaded_image.getbuffer())
    return file_name


def main():
    st.title("Face Detection App")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button('segment'):
        if uploaded_image is not None:
            save_name = save_image(uploaded_image, uploaded_image.name)
            
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            n_faces = detect_face.detect_faces(save_name)
            if n_faces == 1:
                mask_img_path = face_seg.get_segment_mask(save_name)
                st.image(mask_img_path, caption="Face Mask Image", use_column_width=True)
            else:
                st.error(f"No of face detected = {n_faces}, cannot run segmentation")
        else:
            st.error("Missing img file")
if __name__ == "__main__":
    main()