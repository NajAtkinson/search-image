!pip install flask supabase transformers torch torchvision requests numpy scikit-learn Pillow pyngrok
!ngrok authtoken 2mwYQcoBcyZuNO524nVHUqS6OXE_2c76EKemxr4dE5JYS85yC
from pyngrok import ngrok

# Set your ngrok authtoken
ngrok.set_auth_token("2mwYQcoBcyZuNO524nVHUqS6OXE_2c76EKemxr4dE5JYS85yC")  # Replace with your actual authtoken.
!pip install python-dotenv


from flask import Flask, request, jsonify
from pyngrok import ngrok
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Initialize Flask app
app = Flask(__name__)

# Set your ngrok authtoken
ngrok.set_auth_token("2mwYQcoBcyZuNO524nVHUqS6OXE_2c76EKemxr4dE5JYS85yC")

# Supabase configuration
supabase_url = 'https://bpdgwtzndipaybatzcix.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJwZGd3dHpuZGlwYXliYXR6Y2l4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzEyNzkxNDgsImV4cCI6MjA0Njg1NTE0OH0.Jan2VtasYMgsVXiAhbRdlyDC4ol_Ra9mB_eJ3uqUiKE'
supabase = create_client(supabase_url, supabase_key)

# Load the CLIP model and processor (using base version for 768 dimensions)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")





def adjust_embedding_to_1024(embedding, target_dim=1024):
    # Convert embedding to numpy array
    embedding = np.array(embedding)
    # Pad with zeros if necessary
    if embedding.shape[0] < target_dim:
        embedding = np.pad(embedding, (0, target_dim - embedding.shape[0]), mode='constant')
    return embedding.tolist()




def get_image_embedding(image_url):
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Generate embedding
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # Convert to numpy and adjust to 1024 dimensions
        embedding = outputs[0].cpu().numpy().tolist()
        adjusted_embedding = adjust_embedding_to_1024(embedding)

        return adjusted_embedding

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")



@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    data = request.json  # Correctly access JSON data
    user_id = data.get('user_id')
    image_url = data.get('image_url')

    try:
        # Get the image embedding
        embedding = get_image_embedding(image_url)

        # Insert data into Supabase
        response = supabase.table('images').insert({
            'user_id': user_id,
            'image_url': image_url,
            'img_embedding': str(embedding)  # Store as string representation of list
        }).execute()

        return jsonify({
            'message': 'Image uploaded successfully',
            'data': response.data
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-image', methods=['POST'])
def search_image():
    try:
        data = request.json  # Correctly access JSON data without parentheses
        input_image_url = data.get('image_url')

        # Convert threshold to float, with a default value of 0.8
        threshold = float(data.get('threshold', 0.6))

        input_embedding = get_image_embedding(input_image_url)
        if input_embedding is None:
            return jsonify({"error": "Failed to generate embedding for the input image."}), 400

        # Fetch all images and their embeddings from Supabase
        response = supabase.table("images").select("image_url, img_embedding, user_id").execute()

        if not response.data:
            return jsonify({"error": "No images found in the database."}), 404

        images_data = response.data
        stored_embeddings = []
        user_ids = []

        for image in images_data:
            embedding_str = image['img_embedding']
            try:
                embedding = ast.literal_eval(embedding_str)
                stored_embeddings.append(embedding)
                user_ids.append(image['user_id'])
            except Exception as e:
                continue  # Skip this image if embedding parsing fails

        if not stored_embeddings:
            return jsonify({"error": "No valid embeddings found in the database."}), 404

        stored_embeddings_np = np.array(stored_embeddings)
        input_embedding_np = np.array(input_embedding).reshape(1, -1)

        # Calculate similarities using cosine similarity
        similarities = cosine_similarity(input_embedding_np, stored_embeddings_np)[0]

        # Find indices of matching images based on threshold
        matching_indices = np.where(similarities >= threshold)[0]

        if len(matching_indices) == 0:
            return jsonify({"message": "No similar images found."}), 404

        matching_images = [(images_data[i]['image_url'], user_ids[i]) for i in matching_indices]

        return jsonify({"matching_images": matching_images}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    public_url = ngrok.connect(5000)  # Start ngrok tunnel on port 5000
    print("Public URL:", public_url)  # Print the public URL for access
    app.run(port=5000)


# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("Public URL:", public_url)

# Run the Flask app
app.run(port=5000)
