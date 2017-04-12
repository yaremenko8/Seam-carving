import numpy as np
from numpy.linalg import norm
from scipy import misc



def seam_carve(img_file, mode, mask_file = None): #accepts both files and arrays
    VERTICAL = "vertical" in mode
    SHRINK   = "shrink"   in mode
    pic = misc.imread(img_file) if type(img_file) == str else np.copy(img_file)
    #print(pic.shape)
    if mask_file != None:
        mask = misc.imread(mask_file) if type(mask_file) == str else np.copy(mask_file)
    else:
        mask = [[0 for i in range(pic.shape[1])] for j in range(pic.shape[0])]
        
    if VERTICAL:
        pic  = np.transpose(pic, (1, 0, 2))
        mask = np.transpose(mask, (1, 0, 2))
    mask = mask.tolist()

    WIDTH  = pic.shape[0]
    HEIGHT = pic.shape[1]

    bound = lambda x, c: x if 0 < x < c else 0 if x <= 0 else c
    mtx_grad  = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
    seam_mask = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]

        
    for i in range(WIDTH):
        for j in range(HEIGHT):
            Ix = norm(pic[bound(i + 1, WIDTH - 1)][j] - pic[bound(i - 1, WIDTH - 1)][j])
            Iy = norm(pic[i][bound(j + 1, HEIGHT - 1)] - pic[i][bound(j - 1, HEIGHT - 1)])
            mtx_grad[i][j] = norm((Ix, Iy))
            if mask[i][j][1]   == 255:
                mtx_grad[i][j]    =  1000     # I guess
            elif mask[i][j][0] == 255:
                mtx_grad[i][j]    = -1000     # that'll do

    mtx_seam = np.copy(mtx_grad).tolist()
    for i in range(1, WIDTH):
        for j in range(HEIGHT):
            mtx_seam[i][j] += min(mtx_seam[i - 1][bound(j - 1, HEIGHT - 1)], 
                                  mtx_seam[i - 1][bound(j    , HEIGHT - 1)],
                                  mtx_seam[i - 1][bound(j + 1, HEIGHT - 1)])

    pic  = pic.tolist()
    i = WIDTH - 1
    j = np.argmin(mtx_seam[i])
    if SHRINK:
        pic[i].pop(j)
        mask[i].pop(j)
    else:
        temp = (np.array(pic[i][j]) + np.array(pic[i][bound(j + 1, HEIGHT - 1)])) // 2
        pic[i].insert(j + 1, temp)
        mask[i].insert(j + 1, mask[i][j])
    seam_mask[i][j] = 1
    while i > 0:
        offset = np.argmin((mtx_seam[i - 1][bound(j - 1, HEIGHT - 1)], 
                            mtx_seam[i - 1][bound(j    , HEIGHT - 1)],
                            mtx_seam[i - 1][bound(j + 1, HEIGHT - 1)])) - 1
        j = bound(j + offset, HEIGHT - 1)
        i -= 1
        if SHRINK:
            pic[i].pop(j)
            mask[i].pop(j)
        else:
            temp = (np.array(pic[i][j]) + np.array(pic[i][bound(j + 1, HEIGHT - 1)])) // 2
            pic[i].insert(j + 1, temp)
            mask[i].insert(j + 1, mask[i][j])
        seam_mask[i][j] = 1

    pic = np.array(pic, np.dtype('uint8'))
    mask = np.array(mask, np.dtype('uint8'))
    if VERTICAL:
        pic = np.transpose(pic, (1, 0, 2))
        mask = np.transpose(mask, (1, 0, 2))
        seam_mask = np.transpose(seam_mask, (1, 0))
    #print(pic.shape)        
    
    return (pic, mask, seam_mask)
