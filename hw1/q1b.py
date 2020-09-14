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
def q1b_mean(d):
    return q1a_mean()*d
def q1b_var(d):
    return q1a_var()*d

#question 1a answer
print(q1a_mean(), q1a_var())

#question 1b answer
d = 10 
print(q1b_mean(d), q1b_var(d))

