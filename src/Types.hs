module Types where
import Numeric.LinearAlgebra

type RV = Vector R
type RM = Matrix R
type ZVector = Vector Z

type MMP = (RM, RV) -- Matmul params
type MMP' e = (Matrix e, Vector e)

type Inp a = a
type Out a = a
type Tgt a = a

type Two x = (x, x)
type Three x = (x, x, x)
type TwoNOne x = (Two x, x)