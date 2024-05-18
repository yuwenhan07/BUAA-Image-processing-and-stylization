# import os
# import re
# import time
# import argparse
# from flask import Flask, request, render_template, send_from_directory
# import torch
# from torchvision import transforms
# import utils
# from transformer_net import TransformerNet
# from vgg import Vgg16

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# MODEL_FOLDER = '../new_saved_model'
# OUTPUT_FOLDER = 'outputs'

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# if not os.path.exists(OUTPUT_FOLDER):
#     os.makedirs(OUTPUT_FOLDER)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'content_image' not in request.files or 'model' not in request.form:
#         return 'No file part'
    
#     content_image = request.files['content_image']
#     model_name = request.form['model']

#     if content_image.filename == '':
#         return 'No selected file'
    
#     content_image_path = os.path.join(UPLOAD_FOLDER, content_image.filename)
#     content_image.save(content_image_path)
    
#     output_image_path = os.path.join(OUTPUT_FOLDER, 'output_' + content_image.filename)
    
#     args = argparse.Namespace(
#         content_image=content_image_path,
#         content_scale=None,
#         output_image=output_image_path,
#         model=os.path.join(MODEL_FOLDER, model_name),
#         cuda=torch.cuda.is_available(),
#         export_onnx=None
#     )
    
#     stylize(args)
    
#     return send_from_directory(OUTPUT_FOLDER, 'output_' + content_image.filename)

# def stylize(args):
#     device = torch.device("cuda" if args.cuda else "cpu")

#     content_image = utils.load_image(args.content_image, scale=args.content_scale)
#     content_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.mul(255))
#     ])
#     content_image = content_transform(content_image)
#     content_image = content_image.unsqueeze(0).to(device)

#     with torch.no_grad():
#         style_model = TransformerNet()
#         state_dict = torch.load(args.model)
#         for k in list(state_dict.keys()):
#             if re.search(r'in\d+\.running_(mean|var)$', k):
#                 del state_dict[k]
#         style_model.load_state_dict(state_dict)
#         style_model.to(device)
#         style_model.eval()
#         output = style_model(content_image).cpu()
    
#     utils.save_image(args.output_image, output[0])

# if __name__ == "__main__":
#     app.run(debug=True)
import os
import re
import time
import argparse
from flask import Flask, request, render_template, send_from_directory, url_for
import torch
from torchvision import transforms
import utils
from transformer_net import TransformerNet

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = '../new_saved_model'
OUTPUT_FOLDER = 'outputs'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'content_image' not in request.files or 'model' not in request.form:
        return 'No file part'
    
    content_image = request.files['content_image']
    model_name = request.form['model']

    if content_image.filename == '':
        return 'No selected file'
    
    content_image_path = os.path.join(UPLOAD_FOLDER, content_image.filename)
    content_image.save(content_image_path)
    
    output_image_path = os.path.join(OUTPUT_FOLDER, 'output_' + content_image.filename)
    
    args = argparse.Namespace(
        content_image=content_image_path,
        content_scale=None,
        output_image=output_image_path,
        model=os.path.join(MODEL_FOLDER, model_name),
        cuda=torch.cuda.is_available(),
        export_onnx=None
    )
    
    stylize(args)
    
    return render_template('index.html', 
                           original_image=url_for('uploaded_file', filename=content_image.filename, folder=UPLOAD_FOLDER),
                           stylized_image=url_for('uploaded_file', filename='output_' + content_image.filename, folder=OUTPUT_FOLDER))

@app.route('/uploads/<folder>/<filename>')
def uploaded_file(folder, filename):
    return send_from_directory(folder, filename)

def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    
    utils.save_image(args.output_image, output[0])

if __name__ == "__main__":
    app.run(debug=True)
