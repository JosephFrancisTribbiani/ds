import matplotlib
import numpy as np
from pathlib import Path


CLASSNAMES = [
    "10C", "10D", "10H", "10S", 
    "2C", "2D", "2H", "2S", 
    "3C", "3D", "3H", "3S", 
    "4C", "4D", "4H", "4S", 
    "5C", "5D", "5H", "5S", 
    "6C", "6D", "6H", "6S", 
    "7C", "7D", "7H", "7S", 
    "8C", "8D", "8H", "8S", 
    "9C", "9D", "9H", "9S", 
    "AC", "AD", "AH", "AS", 
    "JC", "JD", "JH", "JS", 
    "KC", "KD", "KH", "KS", 
    "QC", "QD", "QH", "QS"
    ]
ROOT = Path(__file__).resolve().parent
LABEL_PATH = ROOT / "labels.txt"
CMAP = "Spectral"


def main() -> None:
    cmap = matplotlib.colormaps.get_cmap(CMAP)
    for cl_idx, c in enumerate(np.linspace(0, 1, 52)):
        color = cmap(c)[:3]
        rgb   = ", ".join(map(lambda val: str(int(val*255)), color))
        line  = '<Label value="%s" background="rgb(%s)"/>\n' % (CLASSNAMES[cl_idx], rgb)
        with open(LABEL_PATH, 'a') as f:
            f.write(line)
    return
    

if __name__ == "__main__":
    main()
