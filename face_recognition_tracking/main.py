# image = image_inference("/assets/images/Akshay Kumar_1.jpg")
from face_recognition_tracking.inference import image_inference

# #
# save_images_embedding(
#     "/Users/gauravneupane/Documents/ml/fastapi_projects/face_recognition_tracking/assets/images"
# )

output = image_inference(
    "/Users/gauravneupane/Documents/ml/fastapi_projects/face_recognition_tracking/assets/images/faces/Akshay Kumar/Akshay Kumar_0.jpg"
)

print(output)
