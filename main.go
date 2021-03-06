package main

import (
	"encoding/csv"
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"log"
	"math"
	"os"
	"strconv"
	"time"
)

var randSource = rand.NewSource(uint64(time.Now().UnixNano()))

func main () {
	//parameters
	modelStdDev := 0.1
	measurementStdDev := 0.3
	const numDataPoints = 10
	T := 1.0  //sampling interval

	//define A, Q, H, R
	A := mat.NewDense(4,4, []float64{1, T, 0, 0, 0, 1, 0, 0, 0, 0, 1, T, 0, 0, 0, 1})
	Q := mat.NewSymDense(4, []float64{math.Pow(T, 3)/3.0, math.Pow(T, 2)/2.0, 0, 0,
											math.Pow(T, 2)/2.0, T, 0, 0,
											0, 0,math.Pow(T, 3)/3.0, math.Pow(T, 2)/2.0,
											0, 0, math.Pow(T, 2)/2.0, T}) //temp
	Q.ScaleSym(math.Pow(modelStdDev, 2), Q)
	H := mat.NewDense(2,4, []float64{1, 0, 0, 0, 0, 0, 1, 0})
	R := scaledId(2, math.Pow(measurementStdDev, 2))

	//generate and store all random data in matrix
	//constData := 100.0
	//dataDist := Gaussian{mean: constData, stdDev: 5}
	data := make([]mat.Matrix, numDataPoints)  //slice of matrices
	noisyData := make([]mat.Matrix, numDataPoints)  //slice of matrices
	x0 := mat.NewDense(4,1, []float64{0, 0.3, 0, .3})  //initial state
	data[0] = x0
	noisyData[0] = x0
	for i := 1; i < numDataPoints; i++ {
		data[i] = getNewState(data[i-1], A, Q) //todo: see if i broke this (always reassigning)
		measuredData := mat.NewDense(2, 1, []float64{data[i].At(0, 0), data[i].At(2, 0)})
		noisyData[i] = getNewState(measuredData, scaledId(2, 1), R) //todo: fix workaround
	}
	//initialize slices to store states and covariances & set initial values
	states := make([]mat.Matrix, numDataPoints)  //slice of matrices
	states[0] = x0

	//initialize variables for innovation, predicted measurement covariance, and Kalman gain
	v := mat.NewDense(2,1, nil)
	S := mat.NewDense(2,2, nil)
	S_inv := mat.NewDense(2,2, nil)
	K := mat.NewDense(4,2, nil)

	modelCovars := make([]mat.Matrix, numDataPoints)  //slice of matrices
	initalModelCovar := 0.0
	modelCovars[0] = mat.NewDense(4,4, getConstList(16, initalModelCovar))

	for i := 1; i < numDataPoints; i++ {
		//predict step
		x_pred := mat.NewDense(4,1, nil)
		x_pred.Mul(A, states[i-1])

		P_pred := mat.NewDense(4,4, nil)
		P_pred.Product(A, modelCovars[i-1], A.T()) //temp
		P_pred.Add(P_pred, Q)

		//update step
		v.Mul(H, x_pred) //temp
		v.Sub(noisyData[i], v)

		S.Product(H, P_pred, H.T()) //temp
		S.Add(S, R)
		S_inv.Inverse(S)

		K.Product(P_pred, H.T(), S_inv)

		x_curr := mat.NewDense(4,1, nil)
		x_curr.Mul(K, v) //temp
		x_curr.Add(x_pred, x_curr)
		states[i] = x_curr

		P_curr := mat.NewDense(4,4, nil)
		P_curr.Product(K, S, K.T()) //temp
		P_curr.Sub(P_pred, P_curr)
		modelCovars[i] = P_curr
	}

	plotKalman(noisyData, data, states)
	exportToCSV(noisyData, data, states)
}

//used to get next state from previous state (for data generation) and noisy state from current state (measurement generation)
func getNewState (x, stateChange mat.Matrix, covariances mat.Symmetric) mat.Matrix {
	normal, _ := distmv.NewNormal(getConstList(numRows(covariances), 0), covariances, randSource)
	normalMat := mat.NewDense(numRows(covariances), numCols(x), normal.Rand(nil))
	nextState := mat.NewDense(numRows(x), numCols(x), nil)
	nextState.Mul(stateChange, x)
	nextState.Add(nextState, normalMat)
	return nextState
}

//plots measurements, actual data, and filter predictions
func plotKalman (noisyData, data, states []mat.Matrix) {
	p, _ := plot.New()

	p.Title.Text = "Kalman Filter Example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	plotutil.AddLinePoints(p,
		"Actual", getPoints(data),
		"Predictions", getPoints(states),
		"Measurements", getPoints(noisyData))


	// Save the plot to a PNG file.
	p.Save(4*vg.Inch, 4*vg.Inch, "points.png")
}

func exportToCSV (noisyData, data, states []mat.Matrix) {
	dataX, dataY := getXYLists(data)
	noisyDataX, noisyDataY := getXYLists(noisyData)
	statesX, statesY := getXYLists(states)

	csvData := make([][]string, len(data)+1)
	csvData[0] = []string{"dataX", "dataY", "measurementX", "measurementY", "predictionX", "predictionY"}
	for i := range data {
		csvData[i+1] = []string{dataX[i], dataY[i], noisyDataX[i], noisyDataY[i], statesX[i], statesY[i]}
	}

	//csvData := [][]string{dataX, dataY, noisyDataX, noisyDataY, statesX, statesY}

	file, _ := os.Create("result.csv")
	defer file.Close()

	w := csv.NewWriter(file)
	w.WriteAll(csvData) // calls Flush internally

	if err := w.Error(); err != nil {
		log.Fatalln("error writing csv:", err)
	}
}

//util for saving data to csv
func getXYLists (matList []mat.Matrix) ([]string, []string) {
	xList := make([]string, len(matList))
	yList := make([]string, len(matList))
	for i := range matList {
		xList[i] = strconv.FormatFloat(matList[i].At(0,0), 'f', -1, 64)
		yList[i] = strconv.FormatFloat(matList[i].At(numRows(matList[i])/2,0), 'f', -1, 64)
	}
	return xList, yList
}

//matrix utils

func printMat (m mat.Matrix) {
	fmt.Printf("%v\n", mat.Formatted(m, mat.Prefix(""), mat.Excerpt(0)))
}

func numRows (m mat.Matrix) int {
	rows, _ := m.Dims()
	return rows
}

func numCols (m mat.Matrix) int {
	_, cols := m.Dims()
	return cols
}

func scaledId (n int, scalar float64) mat.Symmetric {
	m := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		m.SetSym(i, i, 1)
	}
	m.ScaleSym(scalar, m)
	return m
}

func getConstList (n int, value float64) []float64 {
	m := make([]float64, n)
	for i := 0; i < n; i++ {
		m[i] = value
	}
	return m
}


func getPoints (list []mat.Matrix) plotter.XYs {
	pts := make(plotter.XYs, len(list))
	for i := range pts {
		pts[i].X = list[i].At(0,0)
		pts[i].Y = list[i].At(numRows(list[i])/2,0)
	}
	return pts
}