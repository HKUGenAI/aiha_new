import os
import uuid
import hashlib
from azure.storage.blob import BlobServiceClient
from azure.storage.blob._container_client import ContainerClient
from dotenv import load_dotenv
from selenium import webdriver

AZURE_IMAGE_CONTAINER_NAME = "images"
GOOGLE_LEN_BASE_URL = "https://lens.google.com/uploadbyurl?url={}"

class google_lens:
    def __init__(self):
        load_dotenv()
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
        self.base_url = GOOGLE_LEN_BASE_URL
        self.blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        self.container_client = self._create_or_get_container(AZURE_IMAGE_CONTAINER_NAME, self.blob_service_client)
    
    def _create_or_get_container(self, container_name: str, blob_service_client: BlobServiceClient) -> ContainerClient:
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
        return container_client
    
    def _get_16B_image_hash(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        sha256_hash = hashlib.sha256(image_data).hexdigest()
        return sha256_hash[:32]
    
    def _local_image_to_url(self, local_path: str) -> str:
        extension = os.path.splitext(local_path)[1][1:]

        if extension not in ["png", "jpg", "jpeg"]:
            raise Exception("Invalid file type. Only png, jpg, and jpeg are supported.")

        image_hash = self._get_16B_image_hash(local_path)
        blob_name = f"{str(uuid.UUID(image_hash, version=4))}.{extension}"
        blob_client = self.container_client.get_blob_client(blob_name)

        if blob_client.exists():
            print("\nBlob already exists in Azure Storage:\n\t" + local_path)
            return blob_client.url

        print("\nUploading to Azure Storage as blob:\n\t" + local_path)
        
        with open(local_path, "rb") as image_file:
            blob_client.upload_blob(image_file)

        return blob_client.url, blob_name

    def _image_byte_to_url(self, image_byte: bytes) -> str:
        image_hash = hashlib.sha256(image_byte).hexdigest()[:32]
        blob_name = f"{str(uuid.UUID(image_hash, version=4))}.jpg"
        blob_client = self.container_client.get_blob_client(blob_name)

        if blob_client.exists():
            print("\nBlob already exists in Azure Storage:\n\t" + blob_name)
            return blob_client.url

        print("\nUploading to Azure Storage as blob:\n\t" + blob_name)
        blob_client.upload_blob(image_byte)

        return blob_client.url, blob_name
    
    def _check_local_image(self, local_path: str) -> bool:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found at {local_path}")
        return True
    
    def get_image_results(self, local_path: str = None, image_byte: bytes = None ,  num_results: int = 6) -> list:
        if local_path is None and image_byte is None:
            raise Exception("No image provided.")
        
        if local_path is not None:
            self._check_local_image(local_path)
            image_url, image_blob_name = self._local_image_to_url(local_path)
        else:
            image_url, image_blob_name = self._image_byte_to_url(image_byte)

        self.driver.get(self.base_url.format(image_url))
        try:
            lens_result = self.driver.find_element(by="id", value="res")
            a_tags = lens_result.find_elements(by="tag name", value="a")[:num_results]
        except:
            with open('error_page.html', 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            raise Exception("No results found.")
        # print(a_tags)

        results = []
        for a_tag in a_tags:
            try:
                heading = a_tag.find_element(by="css selector", value="div[role = 'heading']").text
                link = a_tag.get_attribute("href")
                results.append({"heading": heading, "link": link})
                print(f"\n{heading}\n{link}")
            except:
                pass
        
        self.container_client.delete_blob(image_blob_name)
        print("\nBlob deleted from Azure Storage.\n\t" + image_blob_name)

        return results
    
    def __del__(self):
        self.driver.quit()
        print("\nDriver quit successfully.")


if __name__ == "__main__":
    gl = google_lens()
    results = gl.get_image_results("zhangxiang.jpg")
    for result in results:
        print(result)
    del gl