"""Contact-sheet frames for human gait review; does not change source video."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();reader=imageio.get_reader(a.video)
fps=reader.get_meta_data()['fps'];canvas=Image.new('RGB',(960,520),'white')
draw=ImageDraw.Draw(canvas)
for i,t in enumerate((5.,5.08,5.16,5.24,5.32,5.4)):
    frame=Image.fromarray(reader.get_data(round(t*fps))).resize((320,240))
    x=(i%3)*320;y=(i//3)*260;canvas.paste(frame,(x,y));draw.text((x+8,y+242),f't = {t:.2f}s',fill='black')
reader.close();canvas.save(a.output)
