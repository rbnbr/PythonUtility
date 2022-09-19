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
    r = ""
    add = False
    for j in range(len(s)):
        if not add:
            if s[j] == c_left:
                add = True
                continue
            else:
                continue
        else:
            if s[j] == c_right:
                break
            else:
                r += s[j]
    return r
