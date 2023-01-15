################################################################################
 # Copyright (C) 2022 Maxim Integrated Products, Inc., All Rights Reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included
 # in all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 # IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 # OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 # OTHER DEALINGS IN THE SOFTWARE.
 #
 # Except as contained in this notice, the name of Maxim Integrated
 # Products, Inc. shall not be used except as stated in the Maxim Integrated
 # Products, Inc. Branding Policy.
 #
 # The mere transfer of this software does not imply any licenses
 # of trade secrets, proprietary technology, copyrights, patents,
 # trademarks, maskwork rights, or any other form of intellectual
 # property whatsoever. Maxim Integrated Products, Inc. retains all
 # ownership rights.
 #
 ###############################################################################
"""
Creates a C file of RGB565 pixel format resized to 320x240 from a JPEG, or 24bit BMP file.
"""
from string import Template
import sys
from PIL import Image, ImageOps

ROWSIZE = 8


# Convert 24 bit RGB to 16 bit 565 format
def to565(pixel):
    """
    convert 24bit to 16bit
    """
    red = pixel[0]
    green = pixel[1]
    blue = pixel[2]
    return ((red & 0x00F8) << 8) | ((green & 0x00FC) << 3) | ((blue & 0x00F8) >> 3)


# Load image from command line
if len(sys.argv) < 2:
    print('Error: No input file!\nUsage:')
    print(' python bmp2c.py image_24bit.bmp [-r][-s][-f][-m]')
    print(' python bmp2c.py image.jpg [-r][-s][-f][-m]')
    print('  -r: rotate 90 degrees')
    print('  -s: rescale to fit the TFT size (320x240)')
    print('  -f: flip vertically')
    print('  -m: mirror horizontally')
    sys.exit()

# Load image from command line
file = sys.argv[1]
img = Image.open(file)
imgname = file

# Info
print(f" Name: {imgname} \n Size: {img.size} \n Format: {img.format}")
# print(f" Mode: {img.mode} \n Info: {img.info}\n")

if img.format != 'BMP':
    print('Converted to BMP')
    img.save("temp.bmp", "BMP")
    img = Image.open("temp.bmp")

if img.mode != 'RGB':
    print('Error: Input file mode should be 24bit RGB')
    sys.exit()

for arg in sys.argv:
    # Rotation needed?
    if arg == '-r':
        img = img.rotate(90)
        print('Image rotated')

    # Scale needed?
    if arg == '-s':
        print('Image resized to (320,240)')
        img = img.resize((320, 240))

    # Flip needed?
    if arg == '-f':
        img = ImageOps.flip(img)
        print('Image flipped')

    # Mirror needed?
    if arg == '-m':
        img = ImageOps.mirror(img)
        print('Image mirrored')

# Make sure dimension is even
img = img.resize((2*(img.size[0]//2), 2*(img.size[1]//2)))

print(f'Processing {img.size} image')
img.save("temp.bmp", "BMP")

if (img.size[0] > 320 or img.size[1] > 240):
    print(f'Error: image size cannot be greater than 320x240: use -s to rescale')
    sys.exit()

# Read image data
imgdata = list(img.getdata())

# Open the template
templatefile = open("template.txt", "r")
template = Template(templatefile.read())

# Build the template parameter list
data = {}
data['imgname'] = imgname
data['imgnamecaps'] = imgname.upper()
data['imglen'] = 2*img.size[0] * img.size[1] + 4  # width and height
data['imgsize'] = \
    ''.join(['    0x%.2X, 0x%.2X, 0x%.2X, 0x%.2X, // size: (%d, %d)' % (
            img.size[0] & 0xFF,
            img.size[0] >> 8 & 0xFF,
            img.size[1] & 0xFF,
            img.size[1] >> 8 & 0xFF,
            img.size[0],
            img.size[1],
            )])
data['imgdata'] = ',\n\t'.join([', '.join(['0x%.2X, 0x%.2X' % (to565(x)
                               & 0xFF, to565(x) >> 8 & 0xFF) for x in
                               imgdata[y:y + ROWSIZE]]) for y in
                               range(0, len(imgdata), ROWSIZE)])

# Open the the text file
outputfile = open("image_rgb565" + ".c", "w")

outputfile.write(template.substitute(data))
outputfile.close()

# Save the resized image
img.save("out.bmp")
