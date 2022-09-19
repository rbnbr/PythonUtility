def remove_except_between_chars(s: str, c_left: str, c_right: str):
    """
    Returns everything between the first occurrence of char c_left and the first following occurrence of
     char c_right from s and returns it.
    Does not modify s.
    :param s:
    :param c_left:
    :param c_right:
    :return:
    """
    first_left = s.find(c_left)
    first_right = s[first_left+1:].find(c_right) + (first_left+1)

    return s[first_left+1:first_right]
