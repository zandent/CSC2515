from scipy import integrate
def q1a_mean():
    y0 = lambda x: x
    y1 = lambda x: x**2
    return 2*integrate.quad(y1,0,1)[0] - 2*(integrate.quad(y0,0,1)[0])**2
def q1a_var():
    y0 = lambda x: x
    y1 = lambda x: x**2
    y2 = lambda x: x**3
    y3 = lambda x: x**4
    return 2*integrate.quad(y3,0,1)[0] - 2*4*integrate.quad(y2,0,1)[0]*integrate.quad(y0,0,1)[0] + 6*integrate.quad(y1,0,1)[0]**2 - q1a_mean()**2

print(q1a_mean(), q1a_var())

