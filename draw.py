from drawille import Canvas
from math import sin, radians
import sys

def array2matrix(pixels, num_rows=28):
    matrix = []
    for i in xrange(0, len(pixels), num_rows):
        row = pixels[i:i+num_rows]
        matrix.append(row)
    return matrix

# n is 1-based, not 0-based
def get_nth_line(filename, n):
    with open(filename,'r') as fh:
        for i,line in enumerate(fh,1):
            if i == n:
                return line

def make_canvas(matrix):
    canvas = Canvas()
    for r,row in enumerate(matrix):
        for c,val in enumerate(row):
            if val > 127:
                canvas.set(c, r)
    return canvas

def print_usage():
    script_name = sys.argv[0]
    print "USAGE:   python {0} filename:line".format(script_name)
    print "NOTE:    Line numbers are 1-based"
    print "EXAMPLE: python {0} data/mnist-train.csv:2000".format(script_name)

def main():
    if len(sys.argv) == 1:
        print_usage()
        exit(1)
    
    filename,n = sys.argv[1].split(':')
    line = get_nth_line(filename,int(n))
    list = map(int, line.split(','))
    
    num = list.pop(0)
    print num
    
    matrix = array2matrix(list)
    canvas = make_canvas(matrix)
    print canvas.frame()

if __name__ == '__main__':
    main()
