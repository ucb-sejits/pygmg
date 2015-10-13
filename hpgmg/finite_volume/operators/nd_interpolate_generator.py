import itertools
import operator
import numpy as np

__author__ = 'Shiv Sundram shivsundram@berkeley.edu U.C. Berkeley, shivsundram@lbl.gov, LBNL'

class NDInterpolateGenerator:
    def __init__(self, pre_scale, degree, dimension):
        self.pre_scale = pre_scale
        self.points = degree+1
        self.dimension=dimension


    def generate_coeffs_array(self):
        order=self.points
        arrayname = "grid"
        weight_array_g=[None for c in range(0,order)]
        if order%2:
            for i in range(0, order/2):
                weight_array_g[i] = "-"+"c"+str(order/2-i)
                weight_array_g[order-i-1]=" "+"c"+str(order/2-i)
            weight_array_g[order/2]= "1"
            weight_array_gr = weight_array_g[:]
            weight_array_gr.reverse()

            weight_array = weight_array_g
            weight_array_r = weight_array_gr
            print weight_array_gr
            return weight_array, weight_array_r

        else: #even number of points #THIS IS NOT READY YET
            print("THIS IS NOT READY YET")

    def generateAlgorithm(self):
        renderdata = []
        order = self.points
        dimension=self.dimension
        weight_array, weight_array_r = self.generate_coeffs_array()
        print weight_array, weight_array_r

        interior_point = tuple([order/2 for _ in range(dimension)])
        print "coordinate of cube center ", interior_point

        def flip(side):
            return 1-side

        #grab points
        print "\ngrab points"
        for coord in itertools.product(range(order),repeat=dimension):
            offset = [c-order/2 for c in coord]
            #print offset
            assignment = 'fc'
            for i in range(dimension):
                assignment += str(coord[i])
            assignment+=" =  source_mesh[source_index + Coord"+str(tuple( offset)) +"]"
            print assignment
            renderdata.append(assignment)

        #interpolate in each dimension
        for interpolated_dims in range(1,dimension+1):
            print
            renderdata.append("\n")
            side=1
            for coord in itertools.product(range(order),repeat=dimension-interpolated_dims):
                suffix = ""
                for c in coord:
                    suffix +=str(c)
                for subcoord in itertools.product(range(2),repeat=interpolated_dims):
                    side=flip(side)
                    prefix=""
                    for s in subcoord:
                        prefix += str(s)
                    index=prefix+suffix
                    if side%2:
                        weights= weight_array
                    else:
                        weights=weight_array_r
                    assignment1 = "f"+prefix+"c"+suffix + " = "
                    for j in range(0, order):
                        new_index = index[:interpolated_dims-1]+str(j)+index[interpolated_dims:]
                        code = new_index[:interpolated_dims-1] + 'c' + new_index[interpolated_dims-1:]
                        assignment1 += weights[j]+"*"+"f"+code+"+"
                    assignment1 = assignment1[:-1] #elimate rogue +
                    print assignment1
                    renderdata.append(assignment1)

        print
        renderdata.append("\n")
        for coord in itertools.product(range(2),repeat=dimension):
            offset = [c-order/2 for c in coord]
            #print offset
            assignment="target_mesh[target_index + Coord"+str(tuple(map(operator.add, interior_point, offset))) +"] = "
            assignment += 'f'
            for i in range(dimension):
                assignment += str(coord[i])
            assignment+="c"
            print assignment
            renderdata.append(assignment)
            break

        return renderdata


    def editFile(self):
        render_data = self.generateAlgorithm()
        tab = '            '

        f = open("interpolation.py", 'r')
        flines = f.readlines()
        f.close()
        f = open("interpolation.py", 'w')
        for line in flines:
            f.write(line)
            if "#special interpolator" in line:
                break
        for line in render_data:
            f.write(tab+line+"\n")
        f.close()



if __name__ == '__main__':
    solver, pre_scale, degree, dimension = None, 1.0, 2, 2
    G = NDInterpolateGenerator(pre_scale, degree, dimension)
    print G.dimension
    G.editFile()

