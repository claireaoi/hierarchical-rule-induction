# -----------------------------------------------------
# ------------TEMPLATES

# CAMPERO TEMPLATES------------------
# 1 F(x) <-- F(X)
# 2 F(x)<---F(Z),F(Z,X)
# 3 F(x,y)<-- F(x,Z),F(Z,Y)
# 4 F(X) <-- F(X,X)
# 5 F(X,Y) <-- F(X,Y)
# 8 F(X,X) <-- F(X)
# 9 F(x,y) <-- F(y,x)
# 10 F(x,y)<---F(X,Z),F(Y,Z)
# 11 F(x,y)<-- F(y,x),F(x)
# 12 F(X) <-- F(X,Z)
# 13 F(X) <-- F(X,Z), F(Z)
# 14 F(X) <-- F(X,Z), F(X,Z)
# 15 F(X,X) <-- F(X,Z), F(X,Z)#TODO CHECK this one
# 16 F(X,Z) <-- F(X,Z), F(X,Z)
# Base set for this model:


# 2. "NEW" TEMPLATES----------------
# if head arity 1, 3 possibilities (actually 4 if take all but for these tasks only 3)
# A00 F(X) <-- F(X,Z),F(Z,X)
# A01 F(X) <-- F(X,Z),F(X,Z)
# A10 F(X) <-- F(Z,X),F(Z,X)

# if head arity 2:
# B00 F(X,Y) <-- F(X,Z),F(Z,Y)
# B01 F(X,Y) <-- F(X,Z),F(Y,Z)
# C00 F(X,Y) <-- F(X,Y),F(Y,X)

# If needed, Extended rules (ie with "+") are with an additional OR, such as:
## A10+ F(X) <-- ( F(Z,X),F(Z,X) ) or F(X,T)
## A00+ F(X) <-- (F(X,Z),F(Z,X)) or F(X,T)
## C00+ F(X,Y) <-- ( F(X,Y),F(Y,X) ) or F(X,Y)

##TEMPLATE SET -4
BASESET_OR4 = {
    "unary": ["A00", "OR1Inv"],
    "binary": ["C00", "B00", "OR2Inv"]
}


##TEMPLATE SET -3
BASESET_OR3 = {
    "unary": ["A00", "OR1"],
    "binary": ["C00", "B00", "OR2", "Inv"]
}

##TEMPLATE SET -2
BASESET_OR12 = {
    "unary": ["A00", "OR1"],
    "binary": ["C00", "B00", "OR2"]
}

##TEMPLATE SET -1
BASESET_OR = {
    "unary": ["A00"],
    "binary": ["C00", "B00", "OR2"]
}
#OR1
#OR2
#Inv
#OR1Inv
#OR2Inv

##TEMPLATE SET 0
BASESET = {
    "unary": ["A00"],
    "binary": ["C00", "B00"]
}

##TEMPLATE SET 1
BASESET_EXTENDED = {
    "unary": ["A00+"],
    "binary": ["C00+", "B00+"]
}

# 101
BASESET_EXTENDED_REC = {
    "unary": ["A00+", "Rec1"],
    "binary": ["C00+", "B00+"]
}

# 102
BASESET_EXTENDED_INV = {
    "unary": ["A00+"],
    "binary": ["C00+", "B00+", "Inv"]
}

# 103
BASESET_EXTENDED_BOTH = {
    "unary": ["A00+", "Rec1"],
    "binary": ["C00+", "B00+", "Inv"]
}

# 111
BASESET_REC = {
    "unary": ["A00", "Rec1"],
    "binary": ["C00", "B00"]
}

# 112
BASESET_INV = {
    "unary": ["A00"],
    "binary": ["C00", "B00", "Inv"]
}

# 113
BASESET_BOTH = {
    "unary": ["A00", "Rec1"],
    "binary": ["C00", "B00", "Inv"]
}

# 114
BASESET_ALL = {
    "unary": ["A00", "Rec1", "OR1"],
    "binary": ["C00", "B00", "Inv", "OR2"]
}

##TEMPLATE SET 2
CAMPERO_BASESET = {
    "unary": [1, 2, 4, 12, 13, 14],
    "binary": [3, 5, 8, 9, 10, 11, 16]
}

##TEMPLATE SET 3
BASESET_ARITHMETIC = {
    "unary": ["A10+", "A10"],
    "binary": ["C00+", "B00+", "B000"]
}

##TEMPLATE SET 4
BASESET_FAMILY = {
    "unary": ["A00+"],
    "binary": ["B00", "C00+", "C00"]
}

##TEMPLATE SET 5
BASESET_GRAPH = {
    "unary": ["A00+"],
    "binary": ["C00+", "B00", "C00"]
}

##TEMPLATE SET 6
CUSTOM_SET = {
    "unary": ["A00+"],
    "binary": ["B00", "C00"]
}

##TEMPLATE SET 66 for Even Succ
CUSTOM_66 = {
    "unary": ["A00+"],
    "binary": ["B00", "Inv"]
}
##TEMPLATE SET 666#mini for EvenSucc juts check no bug
CUSTOM_666 = {
    "unary": ["A10+"],
    "binary": ["B00"]
}

##TEMPLATE PROGRESSIVE: 10
PROGRESSIVE_MINI_SET={
    "unary": ["A"],
    "binary": ["B", "C"]

}
##TEMPLATE PROGRESSIVE: 11
PROGRESSIVE_FULL_SET={
    "unary": ["A+"],
    "binary": ["B+", "C+"]

}

##TEMPLATE PROGRESSIVE: 15
EXTRA={
    "unary": ["A+"],
    "binary": ["B+", "C+"]
}

##TEMPLATE Recursive: 16
REC_SET_1={
    "unary": ["Rec1"],
    "binary": ["B00", "Inv"]
}

##TODO: TEMPLATE Recursive: 17
REC_SET_2={
    "unary": ["A00", "Rec1"],
    "binary": ["B00", "C00"]
}

# 217
REC_SET_3={
    "unary": ["A00", "Rec1"],
    "binary": ["B00", "C00", "OR2", "Inv"]
}

def get_template_set(idx_template_set):
    if idx_template_set == 0:
        TEMPLATE_SET = BASESET
    elif idx_template_set == 1:
        TEMPLATE_SET = BASESET_EXTENDED
    elif idx_template_set == 2:
        TEMPLATE_SET = CAMPERO_BASESET
    elif idx_template_set == 3:
        TEMPLATE_SET = BASESET_ARITHMETIC
    elif idx_template_set == 4:
        TEMPLATE_SET = BASESET_FAMILY
    elif idx_template_set == 5:
        TEMPLATE_SET = BASESET_GRAPH
    elif idx_template_set ==6:
        TEMPLATE_SET = CUSTOM_SET
    elif idx_template_set ==66:
        TEMPLATE_SET = CUSTOM_66
    elif idx_template_set ==666:
        TEMPLATE_SET = CUSTOM_666
    elif idx_template_set ==10:
        TEMPLATE_SET = PROGRESSIVE_MINI_SET
    elif idx_template_set ==11:
        TEMPLATE_SET = PROGRESSIVE_FULL_SET
    elif idx_template_set ==15:
        TEMPLATE_SET = EXTRA
    elif idx_template_set ==-1:
        TEMPLATE_SET = BASESET_OR
    elif idx_template_set ==-2:
        TEMPLATE_SET = BASESET_OR12
    elif idx_template_set ==-3:
        TEMPLATE_SET = BASESET_OR3
    elif idx_template_set ==-4:
        TEMPLATE_SET = BASESET_OR4
    elif idx_template_set ==16:
        TEMPLATE_SET = REC_SET_1
    elif idx_template_set ==17:
        TEMPLATE_SET = REC_SET_2
    elif idx_template_set ==217:
        TEMPLATE_SET = REC_SET_3
    elif idx_template_set ==101:
        TEMPLATE_SET = BASESET_EXTENDED_REC
    elif idx_template_set ==102:
        TEMPLATE_SET = BASESET_EXTENDED_INV
    elif idx_template_set ==103:
        TEMPLATE_SET = BASESET_EXTENDED_BOTH
    elif idx_template_set ==111:
        TEMPLATE_SET = BASESET_REC
    elif idx_template_set ==112:
        TEMPLATE_SET = BASESET_INV
    elif idx_template_set ==113:
        TEMPLATE_SET = BASESET_BOTH
    elif idx_template_set == 114:
        TEMPLATE_SET = BASESET_ALL
    else:
        raise NotImplementedError()
    return TEMPLATE_SET