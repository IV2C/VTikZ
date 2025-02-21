from PIL import Image,ImageChops

def trim(im:Image)->Image:
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else: 
        # Failed to find the borders, convert to "RGB"        
        return trim(im.convert('RGB'))