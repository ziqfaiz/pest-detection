import os

def delete_xml_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xml"):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)


delete_xml_files("/home/haziq/Documents/VIP-Project/Detection/VOC2007/RCNN/dataset/Pest/test")
