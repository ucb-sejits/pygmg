from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Element(object):
    def __init__(
            self,
            true_solution=0.0,
            residual=0.0,
            original_right_hand_side=0.0,
            numerical_solution=0.0,
            cell_centered_coefficient=0.0,
            face_centered_coefficient=0.0,
            inverse_diagonal=0.0,
            inverse_l1_norm_of_each_row=0.0,
            is_valid=False,
    ):
        self.true_solution = true_solution
        self.residual = residual
        self.original_right_hand_side = original_right_hand_side
        self.numerical_solution = numerical_solution
        self.cell_centered_coefficient = cell_centered_coefficient
        self.face_centered_coefficient = face_centered_coefficient
        self.inverse_diagonal = inverse_diagonal
        self.inverse_l1_norm_of_each_row = inverse_l1_norm_of_each_row
        self.is_valid = is_valid

    @property
    def u_true(self):
        return self.true_solution

    @property
    def f_minus_av(self):
        return self.residual

    @property
    def f(self):
        return self.original_right_hand_side

    @property
    def u(self):
        return self.numerical_solution

    @property
    def alpha(self):
        return self.cell_centered_coefficient

    @property
    def beta(self):
        return self.face_centered_coefficient

    @property
    def d_inv(self):
        return self.inverse_diagonal

    @property
    def l1_inv(self):
        return self.inverse_l1_norm_of_each_row