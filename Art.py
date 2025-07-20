from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import requests
from typing import Union, List, Dict, Tuple
from pathlib import Path
import logging
import torch
import gradio as gr

class ImageAnalyzer:
    def __init__(self, 
                 caption_model: str = "Salesforce/blip-image-captioning-base",
                 clip_model: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the image analysis models.
        
        Args:
            caption_model (str): Name of the pre-trained captioning model
            clip_model (str): Name of the pre-trained CLIP model
        """
        # Initialize captioning models
        self.caption_processor = BlipProcessor.from_pretrained(caption_model)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model)
        
        # Initialize CLIP models
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        
        # Default styles to analyze
        self.default_styles = [
            "realism", "anime", "impressionism", "cubism", 
            "digital painting", "sketch", "watercolor", 
            "oil painting", "3D rendering", "photography"
        ]
        
    def load_image(self, image_source: Union[str, Path]) -> Image.Image:
        """
        Load an image from either a local file path or URL.
        
        Args:
            image_source: Path to local image or URL
            
        Returns:
            PIL.Image: Loaded image in RGB format
        """
        try:
            if str(image_source).startswith(('http://', 'https://')):
                response = requests.get(image_source, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                image = Image.open(image_source)
            
            return image.convert('RGB')
        
        except Exception as e:
            logging.error(f"Error loading image from {image_source}: {str(e)}")
            raise
    
    def generate_caption(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image: Path to image, URL, or PIL Image object
            
        Returns:
            str: Generated caption for the image
        """
        try:
            # If image is a string or Path, load it first
            if isinstance(image, (str, Path)):
                image = self.load_image(image)
            
            # Process image and generate caption
            inputs = self.caption_processor(image, return_tensors="pt")
            output = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            raise

    def analyze_style(self, 
                     image: Union[str, Path, Image.Image], 
                     styles: List[str] = None) -> Dict[str, float]:
        """
        Analyze the artistic style of the image using CLIP.
        
        Args:
            image: Path to image, URL, or PIL Image object
            styles: List of style labels to analyze. If None, uses default styles.
            
        Returns:
            Dict[str, float]: Dictionary mapping style names to confidence scores (0-100)
        """
        try:
            # If image is a string or Path, load it first
            if isinstance(image, (str, Path)):
                image = self.load_image(image)
            
            # Use default styles if none provided
            if styles is None:
                styles = self.default_styles
            
            # Process inputs
            text_inputs = self.clip_processor(text=styles, return_tensors="pt", padding=True)
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Get similarity scores
            outputs = self.clip_model(**image_inputs, **text_inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Create dictionary of results
            return {style: (score.item() * 100) for style, score in zip(styles, probs[0])}
            
        except Exception as e:
            logging.error(f"Error analyzing style: {str(e)}")
            raise

    def analyze_image(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        """
        Analyze an image using both BLIP for captioning and CLIP for style analysis.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple[str, Dict[str, float]]: Caption and style scores
        """
        try:
            # Generate caption
            caption = self.generate_caption(image)
            
            # Analyze style
            style_scores = self.analyze_style(image)
            
            # Sort style scores by confidence
            sorted_scores = dict(sorted(style_scores.items(), key=lambda x: x[1], reverse=True))
            
            return caption, sorted_scores
        except Exception as e:
            logging.error(f"Error in image analysis: {str(e)}")
            return "Error analyzing image", {"error": 0.0}

def format_style_scores(scores: Dict[str, float]) -> str:
    """Format style scores into a readable string."""
    return "\n".join([f"{style}: {score:.1f}%" for style, score in scores.items()])

def create_gradio_interface():
    # Initialize the analyzer
    analyzer = ImageAnalyzer()
    
    def process_image(image):
        if image is None:
            return "No image provided", "Please upload an image"
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Analyze image
        caption, style_scores = analyzer.analyze_image(image)
        
        # Format results
        style_analysis = format_style_scores(style_scores)
        
        return caption, style_analysis
    
    # Create and launch the interface
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Textbox(label="Image Caption", lines=2),
            gr.Textbox(label="Style Analysis", lines=10)
        ],
        title="🎨 Art Style & Content Analyzer",
        description="Upload an image to analyze its content and artistic style. The AI will generate a description and identify the artistic styles present in the image.",
        examples=[
            # Add example images here if you have any
        ],
        theme=gr.themes.Soft()
    )
    
    return interface

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create and launch the Gradio interface
        interface = create_gradio_interface()
        interface.launch(share=True)
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
