from scipy import integrate
def expect():
    y0 = lambda x: x
    y1 = lambda x: x**2
    return 2*integrate.quad(y1,0,1)[0] - 2*(integrate.quad(y0,0,1)[0])**2

print(expect())

