module Types where
import Numeric.LinearAlgebra

type RVector = Vector R
type RMatrix = Matrix R
type ZVector = Vector Z

type MMP = (RMatrix, RVector) -- Matmul params
type MMP' e = (Matrix e, Vector e)

type Inp a = a
type Out a = a
type Tgt a = a