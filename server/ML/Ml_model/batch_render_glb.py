import bpy
import os
import math
import glob
import json

# ==== Настройки ====
input_dir = "D:/Project/BarrierGateAI/server/ML/Ml_model/models_glb"         # Папка с .glb файлами
output_base = "D:/Project/BarrierGateAI/server/ML/Ml_model/output_dataset"    # Куда сохранять картинки
json_path = os.path.join(output_base, "dataset.json")
views_per_model = 8                                # Сколько ракурсов на 1 машину
image_resolution = 512                             # Разрешение рендера

dataset = []

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)

def setup_camera_and_light():
    # Камера
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    # Свет
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    light.location = (5, 5, 5)
    bpy.context.collection.objects.link(light)

def render_views(obj_name, output_dir):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = image_resolution
    scene.render.resolution_y = image_resolution

    camera = bpy.context.scene.camera
    image_paths = []
    camera_angles = []

    for i in range(views_per_model):
        angle = i * (2 * math.pi / views_per_model)
        camera.location = (4 * math.cos(angle), 4 * math.sin(angle), 2)
        camera.rotation_euler = (math.radians(75), 0, angle + math.pi)

        img_name = f"{obj_name}_{i}.png"
        img_path = os.path.join(output_dir, img_name)
        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        image_paths.append(os.path.join(os.path.basename(output_dir), img_name))
        camera_angles.append(math.degrees(angle))

    return image_paths, camera_angles

# Главный цикл
for glb_path in glob.glob(os.path.join(input_dir, "*.glb")):
    clear_scene()
    import_glb(glb_path)
    setup_camera_and_light()

    filename = os.path.basename(glb_path).replace(".glb", "")
    brand = filename.split("_")[0].upper()    # Марка авто (первые буквы до "_")
    output_dir = os.path.join(output_base, brand)
    os.makedirs(output_dir, exist_ok=True)

    image_paths, angles = render_views(filename, output_dir)

    dataset.append({
        "brand": brand,
        "model_name": filename,
        "images": [{"path": path, "angle_deg": angle} for path, angle in zip(image_paths, angles)]
    })

# Сохраняем dataset.json
os.makedirs(output_base, exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("✅ Рендер завершён. Dataset сохранён.")
