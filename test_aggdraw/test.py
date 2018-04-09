import aggdraw
from PIL import Image
import numpy as np
'''
draw = Draw("RGB", (500, 500))
pen = Pen("black")
brush = Brush("black")
draw.line((50, 50, 100, 100), pen)
draw.rectangle((50, 150, 100, 200), pen)
draw.rectangle((50, 220, 100, 270), brush)
draw.rectangle((50, 290, 100, 340), brush, pen)
draw.rectangle((50, 360, 100, 410), pen, brush)
draw.ellipse((120, 150, 170, 200), pen)
draw.ellipse((120, 220, 170, 270), brush)
draw.ellipse((120, 290, 170, 340), brush, pen)
draw.ellipse((120, 360, 170, 410), pen, brush)
draw.polygon((190+25, 150, 190, 200, 190+50, 200), pen)
draw.polygon((190+25, 220, 190, 270, 190+50, 270), brush)
draw.polygon((190+25, 290, 190, 340, 190+50, 340), brush, pen)
draw.polygon((190+25, 360, 190, 410, 190+50, 410), pen, brush)
'''
im = Image.open("Screen Shot 2018-02-14 at 4.38.08 PM.png")
d = aggdraw.Draw(im)
p = aggdraw.Pen("blue", 20)
d.line((0, 0, 0, 150), p)
#d.line((0, 500, 500, 0), p)
d.flush()
#d = aggdraw.Draw(im)

#d = aggdraw.Draw("RGB", (800, 600), "black")
im.show()
