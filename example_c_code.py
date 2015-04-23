def velocity_weave(posx, posy, vx, vy, tau, g, c1, c2):
    # calculate force using inline C, much faster than python only
    # posx/y ... list
    # c1/2 ... float
    
    n = len(posx)
    
    code = r'''
        float vg = g*tau;
        
        for(int i = 0; i < n; i++) {
            float Fx0 = 0;
            float Fy0 = 0;
        
            for(int j = 0; j < n; j++) {
                if(j != i) {
                    float px1 = posx[i];
                    float px2 = posx[j];
                    float diffx = px1 - px2;

                    float py1 = posy[i];
                    float py2 = posy[j];
                    float diffy = py1 - py2;
                    
                    float r2 = diffx*diffx + diffy*diffy;

                    float r4 = r2 * r2;
                    float r6 = r4 * r2;
                    float r8 = r6 * r2;
                    float r14 = r8 * r6;
                    
                    Fx0 += c1 * diffx / r14 - c2 * diffx / r8;
                    Fy0 += c1 * diffy / r14 - c2 * diffy / r8;
                }
            }

            float vx0 = vx[i];
            vx[i] = vx0 + tau*Fx0;
            
            float vy0 = vy[i];
            vy[i] = vy0 + tau*Fy0 - vg;
        }
    '''
