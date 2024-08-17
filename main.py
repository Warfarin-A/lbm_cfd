import numpy as np
from matplotlib import pyplot

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def main():
    plotDeltaT = 10
    randomS = 0.001
    Nx = 500
    Ny = 100
    NL = 9 
    Nt = 10000
    Vinit = 1.5
    tau = 0.52

    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    F = np.ones((Ny, Nx, NL), dtype=np.float64) + randomS * np.random.randn(Ny, Nx, NL)
    F[:,:,3] = Vinit
    object = np.full((Ny, Nx), False)
    
    # for y in range(0, Ny):
    #     for x in range(0, Nx):
    #         if (distance(Nx//7.5, 4*(Ny//9), x, y)+distance(Nx//4,5*(Ny//9), x, y) < 120.1):
    #             object [y][x] = True

    for y in range(0, Ny):
        for x in range(0, Nx):
            if x > Nx//5 and x < Nx//4 and y < (Ny//2+10) and y > (Ny//2-10):
                object [y][x] = True

    # for y in range(0, Ny):
    #     for x in range(0, Nx):
    #         if (distance(Nx//4, Ny//2, x, y) < 30) and (x == Nx//4):
    #             object[y][x] = True
    # #         if (distance(Nx//4, Ny//2, x, y) < 10):
    # #             object [y][x] = True

    for it in range(Nt):
        print(it)

        F[:,-1,[6,7,8]] = F[:,-2,[6,7,8]]
        F[:,0,[2,3,4]] = F[:,1,[2,3,4]]
        F[-1,:,[1,2,8]] = F[-2,:,[1,2,8]]
        F[0,:,[4,5,6]] = F[1,:,[4,5,6]]

        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

        bndryF = F[object, :]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

        rho = np.sum(F, 2)
        ux = np.sum(F*cxs, 2)/rho
        uy = np.sum(F*cys, 2)/rho

        F[object,:] = bndryF
        ux[object] = 0
        uy[object] = 0

        Feq = np.zeros(F.shape)

        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:,:,i] = w * rho * (
                1 + 3 * (cx*ux + cy*uy) + (9*(cx*ux + cy*uy)**2)/2 - 3*(ux**2 + uy**2)/2
            )
            
        F = F + -(1/tau) * (F-Feq)
        # vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        # vorticity[object] = np.nan
        
        if (it%plotDeltaT == 0):
            if (it < Nt-1):
                # pyplot.imshow(vorticity, cmap='RdBu')
                pyplot.imshow(np.sqrt(ux**2 + uy**2))
                pyplot.pause(.01)
                pyplot.cla()
        elif (it == Nt-1):
                # vot = pyplot.imshow(vorticity, cmap='RdBu')
                spd = pyplot.imshow(np.sqrt(ux**2 + uy**2))
                pyplot.show()    
               
if __name__ == "__main__":
    main()