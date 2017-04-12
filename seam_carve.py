import numpy as np
from numpy.linalg import norm
from scipy import misc
from math import sqrt



def seam_carve(img_file, mode, mask = None): #accepts both files and arrays
    VERTICAL = "vertical" in mode
    SHRINK   = "shrink"   in mode
    pic = misc.imread(img_file) if type(img_file) == str else np.copy(img_file)
    if mask is not None:
        mask = misc.imread(mask) if type(mask) == str else np.copy(mask)
    else:
        mask = np.array([[0 for i in range(pic.shape[1])] for j in range(pic.shape[0])])
        
    if VERTICAL:
        pic  = np.transpose(pic, (1, 0, 2))
        mask = np.transpose(mask, (1, 0))
    mask = mask.tolist()

    WIDTH  = pic.shape[0]
    HEIGHT = pic.shape[1]


    bound = lambda x, c: x if 0 < x < c else 0 if x <= 0 else c
    mtx_grad  = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
    seam_mask = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
    i_mtx     = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]

    pic  = pic.tolist()

        
    for i in range(WIDTH):
        for j in range(HEIGHT):

            i_mtx[i][j] = 0.299 * pic[i][j][0] + 0.587 * pic[i][j][1] + 0.114 * pic[i][j][2]

    for i in range(WIDTH):
        for j in range(HEIGHT):

            mtx_grad[i][j] =  mask[i][j] * WIDTH * HEIGHT * 256

            Ix = i_mtx[bound(i + 1, WIDTH - 1)][j] - i_mtx[bound(i - 1, WIDTH - 1)][j]
            Iy = i_mtx[i][bound(j + 1, HEIGHT - 1)] - i_mtx[i][bound(j - 1, HEIGHT - 1)]
            mtx_grad[i][j] += sqrt(Ix ** 2 + Iy ** 2)

            if i > 0:
                mtx_grad[i][j] += min(mtx_grad[i - 1][bound(j - 1, HEIGHT - 1)], 
                                      mtx_grad[i - 1][bound(j    , HEIGHT - 1)],
                                      mtx_grad[i - 1][bound(j + 1, HEIGHT - 1)])
    
    i = WIDTH - 1
    j = np.argmin(mtx_grad[i])
    if SHRINK:
        pic[i].pop(j)
        mask[i].pop(j)
    else:
        temp = [(pic[i][j][0] + pic[i][bound(j + 1, HEIGHT - 1)][0]) // 2,
                (pic[i][j][1] + pic[i][bound(j + 1, HEIGHT - 1)][1]) // 2,
                (pic[i][j][2] + pic[i][bound(j + 1, HEIGHT - 1)][2]) // 2]
        pic[i].insert(j + 1, temp)
        mask[i].insert(j + 1, mask[i][j])
    seam_mask[i][j] = 1
    while i > 0:
        offset = np.argmin((mtx_grad[i - 1][bound(j - 1, HEIGHT - 1)], 
                            mtx_grad[i - 1][bound(j    , HEIGHT - 1)],
                            mtx_grad[i - 1][bound(j + 1, HEIGHT - 1)])) - 1
        j = bound(j + offset, HEIGHT - 1)
        i -= 1
        if SHRINK:
            pic[i].pop(j)
            mask[i].pop(j)
        else:
            temp = [(pic[i][j][0] + pic[i][bound(j + 1, HEIGHT - 1)][0]) // 2,
                    (pic[i][j][1] + pic[i][bound(j + 1, HEIGHT - 1)][1]) // 2,
                    (pic[i][j][2] + pic[i][bound(j + 1, HEIGHT - 1)][2]) // 2]
            pic[i].insert(j + 1, temp)
            mask[i].insert(j + 1, mask[i][j])
        seam_mask[i][j] = 1

    pic = np.array(pic, np.dtype('uint8'))
    mask = np.array(mask, np.dtype('uint8'))
    if VERTICAL:
        pic = np.transpose(pic, (1, 0, 2))
        mask = np.transpose(mask, (1, 0))
        seam_mask = np.transpose(seam_mask, (1, 0))      
    
    return (pic, mask, seam_mask)
