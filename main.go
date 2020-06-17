package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"math"
	"math/rand"
)

type Gaussian struct {
	mean float64
	stdDev float64
}

//type State struct {
//	sx float64  //x-position
//	sy float64  //y-position
//	vx float64  //x-velocity
//	vy float64  //y-velocity
//}

//idea: make interactive tool to visualize kalman on different inputs and types of stuff
//allow slider for input std dev, and for the one the filter thinks it is (model and measurement)
//also initial guess/starting position, number of data points, etc.

func main () {
	//generate and store all random data in matrix
	constData := 20.0
	dataDist := Gaussian{mean: constData, stdDev: 5}
	const numDataPoints = 20
	data := make([]mat.Matrix, numDataPoints)  //slice of matrices
	for i := 0; i < numDataPoints; i++ {
		data[i] = mat.NewDense(1,1, []float64{noisyConstant(dataDist)})  //set each data point to randomly distributed value
	}

	//define A, Q, H, R
	modelStdDev := 1.0
	measurementStdDev := 5.0
	A := identity(1)
	Q := scale(identity(1), math.Pow(modelStdDev, 2))
	H := identity(1)
	R := scale(identity(1), math.Pow(measurementStdDev, 2))

	//initialize slices to store states and covariances & set initial values
	states := make([]mat.Matrix, numDataPoints)  //slice of matrices
	initalState := 20.0
	states[0] = mat.NewDense(1,1, []float64{initalState})

	//initialize variables for innovation, predicted measurement covariance, and Kalman gain
	v := mat.NewDense(1,1, nil)
	S := mat.NewDense(1,1, nil)
	S_inv := mat.NewDense(1,1, nil)
	K := mat.NewDense(1,1, nil)

	modelCovars := make([]mat.Matrix, numDataPoints)  //slice of matrices
	initalModelCovar := 0.0
	modelCovars[0] = mat.NewDense(1,1, []float64{initalModelCovar})

	for i := 1; i < numDataPoints; i++ {
		//predict step
		x_pred := mat.NewDense(1,1, nil)
		x_pred.Mul(A, states[i-1])

		P_pred := mat.NewDense(1,1, nil)
		P_pred.Product(A, modelCovars[i-1], A.T()) //temp
		P_pred.Add(P_pred, Q)

		//update step
		v.Mul(H, x_pred) //temp
		v.Sub(data[i], v)

		S.Product(H, P_pred, H.T()) //temp
		S.Add(S, R)
		S_inv.Inverse(S)

		K.Product(P_pred, H.T(), S_inv)

		x_curr := mat.NewDense(1,1, nil)
		x_curr.Mul(K, v) //temp
		x_curr.Add(x_pred, x_curr)
		states[i] = x_curr

		P_curr := mat.NewDense(1,1, nil)
		P_curr.Product(K, S, K.T()) //temp
		P_curr.Sub(P_pred, P_curr)
		modelCovars[i] = P_curr
	}

	plotKalman(data, states, constData)

}

//function to get (constant) data + noise in matrix form
func noisyConstant (dist Gaussian) float64 {
	return rand.NormFloat64() * dist.stdDev + dist.mean
}

func plotKalman (data []mat.Matrix, states []mat.Matrix, constData float64) {
	p, _ := plot.New()

	p.Title.Text = "Kalman Filter Example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	plotutil.AddLinePoints(p,
		"Measurement", getPoints(data),
		"Predictions", getPoints(states),
		"Actual", getPoints(getConstList(len(data), constData)))


	// Save the plot to a PNG file.
	p.Save(4*vg.Inch, 4*vg.Inch, "points.png")
}


//utilities

func printMat (m mat.Matrix) {
	fmt.Printf("%v", mat.Formatted(m, mat.Prefix(""), mat.Excerpt(0)))
}

func numRows (m mat.Matrix) int {
	rows, _ := m.Dims()
	return rows
}

func numCols (m mat.Matrix) int {
	_, cols := m.Dims()
	return cols
}

func identity (n int) mat.Matrix {
	m := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		m.Set(i, i, 1)
	}
	return m
}

func scale (m mat.Matrix, scalar float64) mat.Matrix {
	retMat := mat.NewDense(numRows(m), numCols(m), nil)
	retMat.Scale(scalar, m)
	return retMat
}

func getConstList (len int, value float64) []mat.Matrix {
	list := make([]mat.Matrix, len)
	for i := 0; i < len; i++ {
		list[i] = mat.NewDense(1,1, []float64{value})
	}
	return list
}

func getPoints (list []mat.Matrix) plotter.XYs {
	pts := make(plotter.XYs, len(list))
	for i := range pts {
		pts[i].X = float64(i)
		pts[i].Y = list[i].At(0,0)
	}
	return pts
}