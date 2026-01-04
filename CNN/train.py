# โปรเจกต์คัดแยกวัตถุ 2 ชนิด (กล้อง vs Art Toy) ด้วย CNN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D, Flatten, Dense, Rescaling,
    RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

# ==================== ค่าคงที่ ====================
IMG_SIZE = (128, 128)  # ขนาดภาพที่ใช้ (ปรับจาก 28x28 เป็น 128x128 สำหรับภาพกล้องมือถือ)
BATCH_SIZE = 16        # จำนวนภาพต่อ Batch (ลดลงเพราะข้อมูลน้อย)
EPOCHS = 20            # จำนวนรอบการเทรน
NUM_CLASSES = 2        # จำนวนคลาส (กล้อง, Art Toy)

# ==================== สร้างโมเดล CNN ====================
def create_model():
    """
    สร้างโมเดล CNN แบบ LeNet-5 (Modified)
    
    โครงสร้าง:
    - Input: รับภาพขนาด 128x128x3 (RGB)
    - Data Augmentation: หมุน, พลิก, ซูม เพื่อเพิ่มข้อมูลเทียม
    - Conv2D (32 filters, 5x5) -> MaxPool2D (2x2)
    - Conv2D (64 filters, 5x5) -> MaxPool2D (2x2)
    - Flatten -> Dense (120) -> Dense (84) -> Dense (2, softmax)
    """
    model = Sequential([
        # ปรับขนาดภาพและ Normalize (0-1)
        Input(shape=IMG_SIZE + (3,)),
        Rescaling(1./255),
        
        # Data Augmentation (ช่วยเมื่อข้อมูลน้อย)
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),

        # Convolutional Block 1
        Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        # Convolutional Block 2
        Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        # Flatten และ Fully Connected Layers
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        
        # Output Layer (2 คลาส: กล้อง, Art Toy)
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# ==================== โหลดข้อมูล ====================
def load_data(data_dir):
    """
    โหลดข้อมูลจากโฟลเดอร์ dataset/
    แบ่งเป็น 80% Train, 20% Validation
    """
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"พบคลาส: {class_names}")

    # Prefetch เพื่อเพิ่มประสิทธิภาพ
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

# ==================== แสดงกราฟผลการเทรน ====================
def plot_history(history):
    """สร้างกราฟ Accuracy และ Loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.savefig('training_result.png', dpi=150)
    print("บันทึกกราฟลงไฟล์ training_result.png")
    plt.show()

# ==================== Main ====================
def main():
    DATA_DIR = os.path.join(os.getcwd(), 'dataset')
    
    # ตรวจสอบโฟลเดอร์
    if not os.path.exists(DATA_DIR):
        print(f"ไม่พบโฟลเดอร์ '{DATA_DIR}'")
        print("กรุณาสร้างโฟลเดอร์ 'dataset/arttoy' และ 'dataset/camera' แล้วใส่รูปภาพ")
        return
    
    # ตรวจสอบว่ามีรูปภาพในโฟลเดอร์
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(subdirs) < 2:
        print(f"พบโฟลเดอร์ย่อยไม่ครบ 2 โฟลเดอร์ (พบ: {subdirs})")
        print("กรุณาสร้างโฟลเดอร์ 'arttoy' และ 'camera' ใน dataset/")
        return

    print("กำลังโหลดข้อมูล...")
    train_ds, val_ds, class_names = load_data(DATA_DIR)
    
    print("\nกำลังสร้างโมเดล...")
    model = create_model()
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # ใช้ sparse เพราะ label เป็นตัวเลข
        metrics=['accuracy']
    )
    
    print("\n========== โครงสร้างโมเดล ==========")
    model.summary()
    
    print("\nกำลังเทรนโมเดล...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    print("\nบันทึกโมเดล...")
    model.save('object_classifier.keras')
    print("บันทึกโมเดลลงไฟล์ object_classifier.keras")
    
    # บันทึกชื่อคลาส
    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    plot_history(history)

if __name__ == '__main__':
    main()
