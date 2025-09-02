import cv2
import os

video_Path = "b3.mp4"
output_Folder = "images/bob"

def extract_frames(video_path, output_folder, interval):
    # 如果資料夾不存在，先建立
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 找出資料夾裡最大編號，避免覆蓋
    existing_files = [f for f in os.listdir(output_folder) if f.endswith(".jpg")]
    if existing_files:
        # 把 "frame_00045.jpg" → 45 取出
        existing_numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        start_count = max(existing_numbers) + 1
    else:
        start_count = 0

    vidcap = cv2.VideoCapture(video_path)
    count = start_count
    frame_number = 0

    while True:
        success, image = vidcap.read()
        if not success:
            break

        if frame_number % interval == 0:
            image_path = os.path.join(output_folder, f"frame_{count:05d}.jpg")
            cv2.imwrite(image_path, image)
            print(f"提取第 {frame_number} 幀, 儲存為 {image_path}")
            count += 1

        frame_number += 1

    vidcap.release()
    print(f"提取完成，共儲存了 {count - start_count} 張圖片。")


translate_to_pic = extract_frames(video_Path, output_Folder,8)
