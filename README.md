# 🎨 Art Style & Content Analyzer

An AI-powered tool that analyzes images to generate natural language descriptions and identify artistic styles using BLIP and CLIP models.

## 🌟 Features

- **Image Captioning**: Generates natural language descriptions of images
- **Style Analysis**: Identifies artistic styles present in the image
- **User-Friendly Interface**: Easy-to-use Gradio web interface
- **Multiple Style Detection**: Analyzes various artistic styles including:
  - Realism
  - Anime
  - Impressionism
  - Cubism
  - Digital painting
  - Sketch
  - Watercolor
  - Oil painting
  - 3D rendering
  - Photography

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd art-analyzer
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Run the application:
```bash
python Art.py
```

2. Open your web browser and go to:
```
http://127.0.0.1:7860
```

3. Upload an image and get:
- AI-generated caption
- Style analysis with confidence scores

## 🛠️ Technical Details

- **BLIP Model**: Used for image captioning
- **CLIP Model**: Used for style analysis
- **Gradio**: Provides the web interface
- **PyTorch**: Powers the deep learning models
- **GPU Support**: Automatically utilizes GPU if available

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Your Name - [your.email@example.com]

Project Link: [https://github.com/yourusername/art-analyzer] 