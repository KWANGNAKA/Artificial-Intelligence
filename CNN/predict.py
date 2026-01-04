import tensorflow as tf
import numpy as np
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

# ================== การตั้งค่า (Config) ==================
IMG_SIZE = (128, 128)          # ขนาดภาพต้องตรงกับตอนเทรน
MODEL_PATH = 'object_classifier.keras'  # ชื่อไฟล์โมเดล
INPUT_FOLDER = 'test_images'   # โฟลเดอร์สำหรับเอารูปมาใส่เพื่อทดสอบ
OUTPUT_FOLDER = 'output_results' # โฟลเดอร์ที่จะบันทึกผลลัพธ์

# ================== ฟังก์ชันโหลด Font ==================
def get_font(size=40):
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except:
            font = ImageFont.load_default()
    return font

# ================== ฟังก์ชันทำนายและวาดรูป ==================
def predict_and_save(model, class_names, img_path, save_dir):
    try:
        # 1. โหลดและเตรียมภาพ
        original_img = Image.open(img_path).convert('RGB')
        width, height = original_img.size
        
        # Resize รูปเพื่อส่งเข้าโมเดล
        img_resized = original_img.resize(IMG_SIZE)
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, 0)

        # 2. ทำนายผล
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0]
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        # 3. วาดผลลัพธ์ลงบนภาพเดิม
        annotated_img = original_img.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        # เลือกสี: Art Toy=เขียว, Camera=ส้ม
        if 'art' in predicted_class.lower(): # เช็คชื่อคลาสแบบยืดหยุ่น
            color = (0, 255, 0)     # เขียว
        else:
            color = (255, 165, 0)   # ส้ม
        
        # วาดกรอบสี่เหลี่ยม
        rect_margin = int(min(width, height) * 0.1)
        draw.rectangle(
            [rect_margin, rect_margin, width-rect_margin, height-rect_margin], 
            outline=color, width=5
        )
        
        # เตรียมข้อความ
        font = get_font(int(height * 0.05))
        text = f"{predicted_class}: {confidence:.1f}%"
        
        # คำนวณขนาดพื้นหลังข้อความ
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # วาดพื้นหลังและตัวอักษร
        text_pos = (rect_margin, rect_margin - text_h - 10)
        if text_pos[1] < 0: text_pos = (rect_margin, rect_margin + 10)
        
        draw.rectangle(
            [text_pos[0]-5, text_pos[1]-5, text_pos[0]+text_w+10, text_pos[1]+text_h+5],
            fill=color
        )
        draw.text(text_pos, text, fill=(255, 255, 255), font=font)

        # 4. [ส่วนที่แก้ไข] บันทึกไฟล์แยกตามโฟลเดอร์
        # สร้างโฟลเดอร์ย่อยตามชื่อคลาส (เช่น output_results/camera)
        target_folder = os.path.join(save_dir, predicted_class)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        filename = os.path.basename(img_path)
        save_path = os.path.join(target_folder, f"result_{filename}")
        annotated_img.save(save_path)
        
        return predicted_class, confidence, save_path
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {img_path}: {e}")
        return None, 0, None

# ================== Main Program ==================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🤖 โปรแกรมคัดแยกรูปภาพ (Separate Output Folders)")
    print("="*50)

    # 1. ตรวจสอบไฟล์โมเดล
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ไม่พบไฟล์โมเดล '{MODEL_PATH}'")
        print("👉 กรุณารันไฟล์ train.py เพื่อสร้างโมเดลก่อนครับ")
        exit()

    # 2. เตรียมโฟลเดอร์ Input
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ สร้างโฟลเดอร์ '{INPUT_FOLDER}' ให้แล้ว")
        print("👉 กรุณาเอารูปมาใส่ แล้วรันใหม่ครับ")
        exit()

    # 3. เตรียมโฟลเดอร์ Output (ล้างของเก่าแล้วสร้างใหม่)
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER) # ลบโฟลเดอร์เก่าทิ้งทั้งหมด
    os.makedirs(OUTPUT_FOLDER)
    print(f"✅ ล้างโฟลเดอร์ผลลัพธ์ '{OUTPUT_FOLDER}' เรียบร้อย")

    # 4. อ่านไฟล์รูป
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"❌ ไม่พบรูปภาพในโฟลเดอร์ '{INPUT_FOLDER}'")
        exit()

    print("กำลังโหลดโมเดล...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = ['art_toy', 'camera']

    print(f"\nกำลังประมวลผล {len(image_files)} รูปภาพ...\n")

    counts = {} # ตัวนับจำนวนแต่ละคลาส

    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(INPUT_FOLDER, filename)
        cls, conf, saved_path = predict_and_save(model, class_names, img_path, OUTPUT_FOLDER)
        
        if cls:
            print(f"[{i}/{len(image_files)}] {filename:<15} --> 📂 {cls.upper()} ({conf:.1f}%)")
            counts[cls] = counts.get(cls, 0) + 1

    print("\n" + "="*50)
    print("📊 สรุปผลการคัดแยก:")
    for cls, count in counts.items():
        print(f"   - {cls}: {count} รูป")
    print(f"\n✅ ดูไฟล์แยกตามโฟลเดอร์ได้ที่: {OUTPUT_FOLDER}")
    print("="*50)