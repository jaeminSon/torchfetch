
__all__ = ['ignore_error', 'return_true_if_pass_else_false', 'return_false_if_fail']


def ignore_error(func):
    def fun(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            pass
    return fun


def return_true_if_pass_else_false(func):
    def fun(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return True
        except:
            return False
    return fun


def return_false_if_fail(func):
    def fun(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return False
    return fun
