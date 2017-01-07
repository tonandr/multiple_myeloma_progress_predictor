/**
 * Copyright 2015, 2016 (C) Inwoo Chung (gutomitai@gmail.com)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * 		http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import java.util.Base64;
import java.util.LinkedList;

/**  
Minimize a continuous differentialble multivariate function. Starting point
is given by "X" (D by 1), and the function named in the string "f", must
return a function value and a vector of partial derivatives. The Polack-
Ribiere flavour of conjugate gradients is used to compute search directions,
and a line search using quadratic and cubic polynomial approximations and the
Wolfe-Powell stopping criteria is used together with the slope ratio method
for guessing initial step sizes. Additionally a bunch of checks are made to
make sure that exploration is taking place and that extrapolation will not
be unboundedly large. The "length" gives the length of the run: if it is
positive, it gives the maximum number of line searches, if negative its
absolute gives the maximum allowed number of function evaluations. You can
(optionally) give "length" a second component, which will indicate the
reduction in function value to be expected in the first line-search (defaults
to 1.0). The function returns when either its length is up, or if no further
progress can be made (ie, we are at a minimum, or so close that due to
numerical problems, we cannot get any closer). If the function terminates
within a few iterations, it could be an indication that the function value
and derivatives are not consistent (ie, there may be a bug in the
implementation of your "f" function). The function returns the found
solution "X", a vector of function values "fX" indicating the progress made
and "i" the number of iterations (line searches or function evaluations,
depending on the sign of "length") used.

Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)

See also: checkgrad 

Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13


(C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen

Permission is granted for anyone to copy, use, or modify these
programs and accompanying documents for purposes of research or
education, provided this copyright notice is retained, and note is
made of any changes that have been made.

These programs and documents are distributed without any warranty,
express or implied.  As the programs were written for research
purposes only, they have not been tested to the degree that would be
advisable in any important application.  All use of these programs is
entirely at the user's own risk.

[ml-class] Changes Made:
1) Function name and argument specifications
2) Output display

----------------------------------------------------------------------------
fmincg (octave) developed by Carl Edward Rasmussen is converted into 
NonlinearCGOptimizer (java) by Inwoo Chung since Dec. 23, 2016.
*/

/**
* Nonlinear conjugate gradient optimizer extended from fmincg developed 
* by Carl Edward Rasmussen.
*  
* @author Inwoo Chung (gutomitai@gmail.com)
* @since Dec. 23, 2016
*/
class NonlinearCGOptimizer extends Optimizer {	
		
	/**
	 * 
	 */
	private static final long serialVersionUID = 4990583252368987021L;
	// Constants.
	public static double RHO = 0.01;
	public static double SIG = 0.5;
	public static double INT = 0.1;
	public static double EXT = 3.0;
	public static int MAX = 20;
	public static double RATIO = 100;
		
	public Map<Integer, Matrix> fmincg(ICostFunction iCostFunc
			, int clusterComputingMode
			, int acceleratingComputingMode
			, Matrix X
			, Matrix Y
			, Map<Integer, Matrix> thetas
			, int numIter
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio
			, List<CostFunctionResult> costFunctionResults) {
		
		// Check exception.
		
		// Optimize the cost function.
		int length = numIter;
		CostFunctionResult r = null;
		
		Matrix T = Utility.unroll(thetas); // Important! ??
		int i = 0;
		int is_failed = 0;
		
		CostFunctionResult r1 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		r = r1;
		
		double f1 = r1.J;
		Matrix df1 = Utility.unroll(r1.thetaGrads);
		
		i += length < 0 ? 1 : 0;
		
		Matrix s = Matrix.constArithmeticalMultiply(-1.0, df1);
		double d1 = -1.0 * Matrix.innerProduct(s.unrolledVector(), s.unrolledVector());
		double z1 = 1.0 / (1.0 - d1);
		
		int success = 0;
		
		double f2 = 0.0;
		Matrix df2 = null;
		double z2 = 0.0;			
		double d2 = 0.0;
		double f3 = 0.0;
		double d3 = 0.0;
		double z3 = 0.0;
		
		Matrix T0 = null;
		double f0 = 0;
		Matrix df0 = null;
		
		while (i < Math.abs(length)) {
			i += length > 0 ? 1 : 0;
			
			T0 = T.clone();
			f0 = f1;
			df0 = df1.clone();
			
			T = T.plus(Matrix.constArithmeticalMultiply(z1, s));
			
			CostFunctionResult r2 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, Utility.roll(T, thetas)
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			r = r2;
			
			f2 = r2.J;
			df2 = Utility.unroll(r2.thetaGrads);
			z2 = 0.0;
			
			i += length < 0 ? 1 : 0;
			
			d2 = Matrix.innerProduct(df2.unrolledVector(), s.unrolledVector());
			f3 = f1;
			d3 = d1;
			z3 = -1.0 * z1;
			
			int M = 0;
			
			if (length > 0) {
				M = MAX;
			} else {
				M = Math.min(MAX, -1 * length - i);
			}
			
			double limit = -1.0;
			
			while (true) {
				while ((f2 > (f1 + z1 * RHO * d1) || (d2 > -1.0 * SIG * d1)) && (M > 0)) {
					limit = z1;
					
					if (f2 > f1) {
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 -f3);
					} else {
						double A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);
						double B = 3.0 * (f3 - f2) -z3 * (d3 + 2 * d2);
						z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
					}
					
					if (Double.isNaN(z2) || Double.isInfinite(z2)) {
						z2 = z3 / 2.0;
					}
					
					z2 = Math.max(Math.min(z2, INT * z3), (1.0 - INT) * z3);
					z1 = z1 + z2;
					T = T.plus(Matrix.constArithmeticalMultiply(z2, s));
					
					CostFunctionResult r3 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio) 
							: iCostFunc.costFunctionR(clusterComputingMode
									, acceleratingComputingMode
									, X
									, Y
									, Utility.roll(T, thetas)
									, lambda
									, isGradientChecking
									, JEstimationFlag
									, JEstimationRatio);
					r = r3;
					
					f2 = r3.J;
					df2 = Utility.unroll(r3.thetaGrads);
					
					M = M - 1;
					
					i += length < 0 ? 1 : 0;
					
					d2 = Matrix.innerProduct(df2.unrolledVector(), s.unrolledVector());
					z3 = z3 - z2;
				}
				
				if (f2 > (f1 + z1 * RHO * d1) || d2 > -1.0 *SIG * d1) 
					break;
				else if (d2 > SIG * d1) {
					success = 1;
					break;
				} else if (M == 0) 
					break;
				
				double A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);
				double B = 3.0 * (f3 - f2) -z3 * (d3 + 2 * d2);
				z2 = -1.0 * d2 * z3 * z3 /(B + Math.sqrt(B * B - A * d2 * z3 * z3));
				
				if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0.0) {
					if (limit < -0.5) {
						z2 = z1 * (EXT - 1.0);
					} else {
						z2 = (limit - z1) / 2.0;
					}
				} else if (limit > -0.5 && ((z2 + z1) > limit)) {
					z2 = (limit - z1) / 2.0;
				} else if (limit < -0.5 && ((z2 + z1) > (z1 * EXT))) {
					z2 = z1 * (EXT - 1.0);
				} else if (z2 < -1.0 * z3 * INT) {
					z2 = -1.0 * z3 * INT;
				} else if (limit > -0.5 && (z2 < (limit - z1) * (1.0 - INT))) {
					z2 = (limit - z1) * (1.0 - INT);
				}
				
				f3 = f2;
				d3 = d2;
				z3 = -1.0 * z2;
				z1 = z1 + z2;
				T = T.plus(Matrix.constArithmeticalMultiply(z2, s));
				
				CostFunctionResult r4 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio) 
						: iCostFunc.costFunctionR(clusterComputingMode
								, acceleratingComputingMode
								, X
								, Y
								, Utility.roll(T, thetas)
								, lambda
								, isGradientChecking
								, JEstimationFlag
								, JEstimationRatio);
				r = r4;
				
				f2 = r4.J;
				df2 = Utility.unroll(r4.thetaGrads);
				
				M = M - 1;
				
				i += length < 0 ? 1 : 0;
				
				d2 = Matrix.innerProduct(df2.unrolledVector(), s.unrolledVector());
			}
			
			if (success == 1) {
				costFunctionResults.add(r); //?
				f1 = f2;
				s = Matrix.constArithmeticalMultiply((Matrix.innerProduct(df2.unrolledVector(), df2.unrolledVector()) 
						- Matrix.innerProduct(df1.unrolledVector(), df2.unrolledVector())) 
						/ (Matrix.innerProduct(df1.unrolledVector(), df1.unrolledVector())), s).minus(df2);
				Matrix tmp = df1.clone();
				df1 = df2.clone();
				df2 = tmp;
				d2 = Matrix.innerProduct(df1.unrolledVector(), s.unrolledVector());
				
				if (d2 > 0) {
					s = Matrix.constArithmeticalMultiply(-1.0, df1);
					d2 = -1.0 * Matrix.innerProduct(s.unrolledVector(), s.unrolledVector());
				}
				
				z1 = z1 * Math.min(RATIO, d1 / (d2 - Double.MIN_VALUE));
				d1 = d2;
				is_failed = 0;
			} else {
				T = T0.clone();
				f1 = f0;
				df1 = df0.clone();
				
				if (!(is_failed == 0 || i > Math.abs(length))) {
					Matrix tmp = df1.clone();
					df1 = df2.clone();
					df2 = tmp;
					s = Matrix.constArithmeticalMultiply(-1.0, df1);
					d1 = -1.0 * Matrix.innerProduct(s.unrolledVector(), s.unrolledVector());
					z1 = 1.0 / (1.0 - d1);
					is_failed = 1;
				}
			}
		}
	
		return Utility.roll(T, thetas);
	}
}

class CostFunctionResult {

	/** Cost function value. */
	public double J;
	
	/** Theta gradient matrix. */
	public Map<Integer, Matrix> thetaGrads = new HashMap<Integer, Matrix>();
	
	/** Estimated theta gradient matrix. */
	public Map<Integer, Matrix> eThetaGrads = new HashMap<Integer, Matrix>();
}

class LBFGSOptimizer extends Optimizer {

	/**
	 * Minimize.
	 * @param iCostFunc
	 * @param sc
	 * @param clusterComputingMode
	 * @param acceleratingComputingMode
	 * @param X
	 * @param Y
	 * @param thetas
	 * @param lambda
	 * @param isGradientChecking
	 * @param JEstimationFlag
	 * @param JEstimationRatio
	 * @param costFunctionResults
	 * @return 
	 */
	public Map<Integer, Matrix> minimize(ICostFunction iCostFunc
			, int clusterComputingMode
			, int acceleratingComputingMode
			, Matrix X
			, Matrix Y
			, Map<Integer, Matrix> thetas
			, int maxIter
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio
			, List<CostFunctionResult> costFunctionResults) {
		
		// Input parameters are assumed to be valid.
		
		// Conduct optimization.		
		// Calculate the initial gradient and inverse Hessian.
		Matrix T0 = Utility.unroll(thetas);
		CostFunctionResult r1 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T0, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T0, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		Matrix G0 = Utility.unroll(r1.thetaGrads);
		Matrix H0 = Matrix.getIdentity(G0.rowLength());
		Matrix P0 = Matrix.constArithmeticalMultiply(-1.0, H0.multiply(G0));
		
		// Calculate an optimized step size.
		double alpha = backtrackingLineSearch(T0
				, G0
				, P0
				, iCostFunc
				, clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, thetas
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio);
		
		// Update the theta.
		Matrix T1 = T0.plus(Matrix.constArithmeticalMultiply(alpha, P0));
		
		CostFunctionResult r2 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T1, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T1, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		
		Matrix G1 = Utility.unroll(r2.thetaGrads);
		costFunctionResults.add(r2);
		
		// Check exception. (Bug??)	
		if (Double.isNaN(r2.J)) {
			return Utility.roll(T0, thetas);
		}
		
		// Store values for inverse Hessian updating.
		LinkedList<Matrix> ss = new LinkedList<Matrix>();
		LinkedList<Matrix> ys = new LinkedList<Matrix>();
		
		Matrix s = T1.minus(T0);
		Matrix y = G1.minus(G0);
		
		ss.add(s);
		ys.add(y);
		
		int count = 1;
		
		while (count <= maxIter) {
			
			// Calculate the next theta, gradient and inverse Hessian.
			T0 = T1;
			
			r1 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, Utility.roll(T0, thetas)
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T0, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			G0 = Utility.unroll(r1.thetaGrads);
			
			// Calculate a search direction.
			P0 = calSearchDirection(ss, ys, G0);
			
			// Calculate an optimized step size.
			alpha = backtrackingLineSearch(T0
					, G0
					, P0
					, iCostFunc
					, clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, thetas
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio);
			
			// Update the theta.
			T1 = T0.plus(Matrix.constArithmeticalMultiply(alpha, P0));
			
			r2 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, Utility.roll(T1, thetas)
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T1, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			
			G1 = Utility.unroll(r2.thetaGrads);
			costFunctionResults.add(r2);
			
			// Check exception. (Bug??)	
			if (Double.isNaN(r2.J)) {
				return Utility.roll(T0, thetas);
			}
			
			// Store values for inverse Hessian updating.
			if (ss.size() == 30) {
				ss.removeFirst();
				ys.removeFirst();
				ss.add(T1.minus(T0));
				ys.add(G1.minus(G0));
			} else {
				ss.add(T1.minus(T0));
				ys.add(G1.minus(G0));
			}

			count++;
		}
		
		return Utility.roll(T1, thetas);
	}
	
	// Calculate a search direction.
	private Matrix calSearchDirection(LinkedList<Matrix> ss, LinkedList<Matrix> ys, Matrix G0) {
		
		// Input parameters are assumed to be valid.
		
		// Calculate P.
		Matrix Q = G0.clone();
		LinkedList<Double> as = new LinkedList<Double>();
		
		for (int i = ss.size() - 1; i >= 0; i--) {
			double rho = 1.0 / ys.get(i).transpose().multiply(ss.get(i)).getVal(1, 1); // Inner product.
			double a = rho * ss.get(i).transpose().multiply(Q).getVal(1, 1);
			as.add(a);
			Q = Q.minus(Matrix.constArithmeticalMultiply(a, ys.get(i)));
		}
		
		double w = 1.0; //ys.getLast().transpose().multiply(ys.getLast()).getVal(1, 1) 
				// / ys.getLast().transpose().multiply(ss.getLast()).getVal(1, 1);
		Matrix H0 = Matrix.constArithmeticalMultiply(w, Matrix.getIdentity(G0.rowLength())); //?
		Matrix R = H0.multiply(Q);
		
		for (int i = 0 ; i < ss.size(); i++) {
			double rho = 1.0 / ys.get(i).transpose().multiply(ss.get(i)).getVal(1, 1);
			double b = Matrix.constArithmeticalMultiply(rho, ys.get(i).transpose().multiply(R)).getVal(1, 1);
			R = R.plus(Matrix.constArithmeticalMultiply(ss.get(i), as.get(as.size() - i - 1) - b));
		}
		
		return Matrix.constArithmeticalMultiply(-1.0, R);
	}
	
	// Conduct backtracking line search.
	private double backtrackingLineSearch(Matrix T
			, Matrix G
			, Matrix P
			, ICostFunction iCostFunc
			, int clusterComputingMode
			, int acceleratingComputingMode
			, Matrix X
			, Matrix Y
			, Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio) {
		
		// Input parameters are assumed to be valid.
		
		// Calculate an optimized step size.
		double alpha = 1.0;
		double c = 0.5;
		double gamma = 0.5;
		
		CostFunctionResult r1 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		
		Matrix T1 = T.plus(Matrix.constArithmeticalMultiply(alpha, P));
		
		CostFunctionResult r2 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T1, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T1, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		
		Matrix M = P.transpose().multiply(G);
		double m = M.getVal(1, 1);
		
		// Check the Armijo-Goldstein condition.
		while (r1.J - r2.J < -1.0 * alpha * c * m) {
			alpha = gamma * alpha;
			
			T1 = T.plus(Matrix.constArithmeticalMultiply(alpha, P));
			
			r2 = classRegType == 0 ? iCostFunc.costFunctionC(clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, Utility.roll(T1, thetas)
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T1, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			
			M = P.transpose().multiply(G);
			m = M.getVal(1, 1);
		} // Close condition?
		
		return alpha;
	}
}

interface ICostFunction {
	public CostFunctionResult costFunctionC(int clusterComputingMode
			, int acceratingComputingMode
			, Matrix X
			, Matrix Y
			, final Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio);
	
	public CostFunctionResult costFunctionR(int clusterComputingMode
			, int acceratingComputingMode
			, Matrix X
			, Matrix Y
			, final Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio);
}

abstract class Optimizer {
	public int classRegType = 0;
}

class Utility {

	/**
	 * Unroll a matrix map.
	 * @param matrixMap a matrix map.
	 * @return Unrolled matrix.
	 */
	public static Matrix unroll(Map<Integer, Matrix> matrixMap) {
		
		// Check exception.
		if (matrixMap == null)
			throw new NullPointerException();
		
		if (matrixMap.isEmpty())
			throw new IllegalArgumentException();
		
		// Unroll.
		int count = 0;
		Matrix unrolledM = null;
		
		for (int i : matrixMap.keySet()) {
			if (count == 0) {
				double[] unrolledVec = matrixMap.get(i).unrolledVector();
				unrolledM = new Matrix(unrolledVec.length, 1, unrolledVec);
				count++;
			} else {
				double[] unrolledVec = matrixMap.get(i).unrolledVector();
				unrolledM = unrolledM.verticalAdd(new Matrix(unrolledVec.length, 1, unrolledVec));
			}
		}
		
		return unrolledM;
	}
	
	/**
	 * Unroll a Matrix.
	 * @param unrolledM
	 * @param matrixMapModel
	 * @return
	 */
	public static Map<Integer, Matrix> roll(Matrix unrolledM
			, Map<Integer, Matrix> matrixMapModel) {
		
		// Check exception.
		if (unrolledM == null || matrixMapModel == null) 
			throw new NullPointerException();
		
		// Input parameters are assumed to be valid.
		
		// Roll.
		Map<Integer, Matrix> matrixMap = new HashMap<Integer, Matrix>();
		int rowIndexStart = 1;
		
		for (int i : matrixMapModel.keySet()) {
			int[] range = {rowIndexStart
			               , rowIndexStart + matrixMapModel.get(i).rowLength() 
			               * matrixMapModel.get(i).colLength() - 1
			               , 1
			               , 1};
			double[] pVec = unrolledM.getSubMatrix(range).unrolledVector();
			Matrix pM = new Matrix(matrixMapModel.get(i).rowLength()
					, matrixMapModel.get(i).colLength(), pVec);
			
			matrixMap.put(i, pM);
			rowIndexStart = rowIndexStart + matrixMapModel.get(i).rowLength() 
					* matrixMapModel.get(i).colLength();
		}
		
		return matrixMap;
	}
	
	/**
	 * Encode NN params. to Base64 string.
	 * @param nn
	 * @return
	 */
	public static String encodeNNParamsToBase64(AbstractNeuralNetwork nn) {
		
		// Check exception.
		if (nn == null) 
			throw new NullPointerException();
		
		// Unroll a theta matrix map.
		double[] unrolledTheta = unroll(nn.thetas).unrolledVector();
		
		// Convert the unrolled theta into a byte array.
		byte[] unrolledByteTheta = new byte[unrolledTheta.length * 8];
		
		for (int i = 0; i < unrolledTheta.length; i++) {
			
			// Convert a double value into a 8 byte long format value.
			long lv = Double.doubleToLongBits(unrolledTheta[i]);
			
			for (int j = 0; j < 8; j++) {
				unrolledByteTheta[j + i * 8] = (byte)((lv >> ((7 - j) * 8)) & 0xff);
			}
		}
		
		// Encode the unrolledByteTheta into the base64 string.
		String base64Str = Base64.getEncoder().encodeToString(unrolledByteTheta);
		
		return base64Str;
	}
	
	/**
	 * Decode Base64 string into a theta map of NN.
	 * @param nn
	 * @return
	 */
	public static void decodeNNParamsToBase64(String encodedBase64NNParams, AbstractNeuralNetwork nn) {
		
		// Check exception.
		if (nn == null && encodedBase64NNParams == null) 
			throw new NullPointerException();
		
		// Input parameters are assumed to be valid.
		// An NN theta model must be configured.
		
		// Decode the encoded base64 NN params. string into a byte array.
		byte[] unrolledByteTheta = Base64.getDecoder().decode(encodedBase64NNParams);
		
		// Convert the byte array into an unrolled theta.
		double[] unrolledTheta = new double[unrolledByteTheta.length / 8];
				
		for (int i = 0; i < unrolledTheta.length; i++) {
			
			// Convert 8 bytes into a 8 byte long format value.
			long lv = 0;
			
			for (int j = 0; j < 8; j++) {
				long temp = ((long)(unrolledByteTheta[j + i * 8])) << ((7 - j) * 8);
				lv = lv | temp;
			}
			
			unrolledTheta[i] = Double.longBitsToDouble(lv);
		}
		
		// Convert the unrolled theta into a NN theta map.
		nn.thetas = roll(new Matrix(unrolledTheta.length, 1, unrolledTheta), nn.thetas);
	}
}

class NeuralNetworkRegression extends AbstractNeuralNetwork {

	public NeuralNetworkRegression(int clusterComputingMode, int acceleratingComputingMode,
			int numLayers, int[] numActs, Optimizer optimizer) {
		super(REGRESSION_TYPE, clusterComputingMode, acceleratingComputingMode, numLayers, numActs, optimizer);
	}
	
	/**
	 * Predict.
	 * @param Matrix of input values for prediction.
	 * @return Predicted result matrix.
	 */
	public Matrix predict(Matrix X) {
		
		// Check exception.
		// Null.
		if (X == null) 
			throw new NullPointerException();
		
		// X dimension.
		if (X.colLength() < 1 || X.rowLength() != numActs[0]) 
			throw new IllegalArgumentException();
		
		return feedForwardR(X);
	}
}

class Matrix{

	/** Matrix number for parallel processing of Apache Spark. */
	public int index;
	
	// Matrix values.
	protected double[][] m;

	public class GaussElimination {

		public final static String TAG = "GaussElimination";
		private final boolean DEBUG = true;
		
		/**
		 *  Constructor.
		 */
		public GaussElimination() {
		}

		/**
		 * <p>
		 * Solve a linear system.
		 * </p>
		 * @param augMatrix
		 * @return Solution.
		 */
		public double[] solveLinearSystem(double[][] augMatrix) {
			
			// Check exception.
			if(augMatrix == null) 
				throw new IllegalArgumentException(TAG + 
						": Can't solve a linear system " +
						"because an input augmented matrix is null.");
			
			int rowSize = augMatrix.length;
			int columnSize = augMatrix[0].length;
			
			for(int i=1; i < rowSize; i++) {
				
				if(columnSize != augMatrix[i].length) 
					throw new IllegalArgumentException(TAG + 
							": Can't solve a linear system " +
							"because an input augmented matrix isn't valid.");
			}
			
			if(!(((rowSize >= 1) && (columnSize >= 2)) 
					&& (rowSize + 1 == columnSize))) 
				throw new IllegalArgumentException(TAG + 
						": Can't solve a linear system " +
						"because an input augmented matrix isn't valid.");
					
			// Solve an input linear system with the Gauss elimination method.
			double[] solution = new double[rowSize];
			
			/*
			 * Make echelon form for the input linear system 
			 * relevant augmented matrix.
			 */
			for(int i = 1; i <= rowSize - 1; i++) {
				
				// Sort equation vectors.
//				sortEquationVectors(augMatrix, i);
				
				// Make zero coefficient.
				makeZeroCoeff(augMatrix, i);
				
				// Check whether it is possible to have only solution.
				if(!checkSolution(augMatrix, i))
					return null;
			}

			// Solve the linear system via back substitution.
			for(int i = rowSize; i >= 1; i--) {
				
				if(augMatrix[i - 1][i - 1] == 0.0) 
					return null;
				
				solution[i - 1] = 
						augMatrix[i - 1][columnSize - 1]/augMatrix[i - 1][i - 1];
				
				for(int j = rowSize; j > i; j--) {
					
					solution[i - 1] -= solution[j - 1]*
							augMatrix[i - 1][j - 1]/augMatrix[i - 1][i - 1];  
				}
			}
			
			return solution;
		}
		
		// Sort equation vectors.
		private void sortEquationVectors(double[][] augMatrix, int colNum) {
			
			// Assume that input parameters are valid. ??
			int rowSize = augMatrix.length;
			int columnSize = augMatrix[0].length;
			double[] tempVector = null;
			
			// Sort the linear system.
			for(int i=colNum; i < rowSize; i++) {
				
				if(Math.abs(augMatrix[i - 1][colNum - 1]) < 
						Math.abs(augMatrix[i][colNum - 1])) {
					
					tempVector = augMatrix[i - 1];
					augMatrix[i - 1] = augMatrix[i];
					augMatrix[i] = tempVector;
				}
			}
		}
		
		// Make zero coefficient.
		private void makeZeroCoeff(double[][] augMatrix, int colNum) {
			
			// Assume that input parameters are valid. ??
			int rowSize = augMatrix.length;
			int columnSize = augMatrix[0].length;
			
			/*
			 * Make coefficient of vectors for input column be zero except 
			 * a pivot vector.
			 */
			double[] pivotVector = augMatrix[colNum - 1];
			
			for(int i=colNum; i < rowSize; i++) {
				
				if(augMatrix[i][colNum - 1] == 0.0)
					continue;
				
				double refCoeff = augMatrix[i][colNum - 1];
				
				for(int j = colNum ; j <= columnSize; j++ ) {
					
					augMatrix[i][j - 1] = pivotVector[colNum - 1]*augMatrix[i][j - 1]
							- refCoeff*pivotVector[j - 1];
				}
			}
		}
		
		// Check whether it is possible to have only solution.
		private boolean checkSolution(double[][] augMatrix, int colNum) {
			
			// Assume that input parameters are valid. ??
			
			int rowSize = augMatrix.length;
			int columnSize = augMatrix[0].length;
			
			// Check.
			boolean isThereCoeff = true;
			
			for(int i = colNum; i < rowSize; i++) {
				
				if(augMatrix[i][colNum] == 0.0) {
					
					isThereCoeff = false;
					break;
				}	
			}
			
			return isThereCoeff;
		}
	}

	/**
	 * Constructor to create a homogeneous matrix.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param val Homogeneous value.
	 */
	public Matrix(int rows, int cols, double val) {
		
		// Create a matrix with a homogeneous value.
		createHomoMatrix(rows, cols, val);
	}
	
	/**
	 * Constructor to create a matrix from a vector.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param v Unrolled values' vector.
	 */
	public Matrix(int rows, int cols, double[] v) {
		
		// Create a matrix.
		createMatrixFromVector(rows, cols, v);
	}
	
	/**
	 * Create a homogeneous matrix.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param val Homogeneous value.
	 */
	public void createHomoMatrix(int rows, int cols, double val) {
		
		// Check exception.
		// Parameters.
		if (rows < 1 || cols < 1) {
			throw new IllegalArgumentException();
		}
		
		// Create a matrix.
		m = new double[rows][];
		
		for (int i = 0; i < rows; i++) {
			m[i] = new double[cols];
			
			for (int j = 0; j < cols; j++) {
				m[i][j] = val;
			}
		}
	}
	
	/**
	 * Create a matrix from a vector.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param v Unrolled values' vector.
	 */
	public void createMatrixFromVector(int rows, int cols, double[] v) {
	
		// Check exception.
		// Parameters.
		if (rows < 1 || cols < 1 || v == null) {
			throw new IllegalArgumentException();
		}
		
		// Valid transformation from a vector to a matrix.
		if (rows * cols != v.length) {
			throw new IllegalArgumentException();
		}
		
		// Create a matrix.
		m = new double[rows][];
		int vIndex = 0;
		
		for (int i = 0; i < rows; i++) {
			m[i] = new double[cols];
			
			for (int j = 0; j < cols; j++) {
				m[i][j] = v[vIndex++];
			}
		}
	}
	
	/**
	 * Get a value of the matrix.
	 * @param rows One-based row index.
	 * @param cols One-based column index.
	 * @return Matrix element value.
	 */
	public double getVal(int rows, int cols) { // One-based index.
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (rows < 1 || rows > m.length || cols < 1 || cols > m[0].length) 
			throw new IllegalArgumentException();
		
		return m[rows - 1][cols - 1];
	}
	
	/**
	 * Set a value of the matrix.
	 * @param rows One-based row index.
	 * @param cols One-based column index.
	 */
	public void setVal(int rows, int cols, double val) { // One-based index.
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (rows < 1 || rows > m.length || cols < 1 || cols > m[0].length) 
			throw new IllegalArgumentException();
		
		m[rows - 1][cols - 1] = val;
	}
	
	/**
	 * Row length of the matrix.
	 * @return Row length.
	 */
	public int rowLength() {
		
		// Check exception.
		checkMatrix();
		
		return m.length;
	}
	
	/**
	 * Column length of the matrix.
	 * @return Column length.
	 */
	public int colLength() {
		
		// Check exception.
		checkMatrix();
				
		return m[0].length;
	}
	
	/**
	 * Get an unrolled vector.
	 * @return Unrolled vector.
	 */
	public double[] unrolledVector() {
		
		// Check exception.
		checkMatrix();
		
		// Create a unrolled vector.
		double[] v = new double[rowLength() * colLength()];
		int vIndex = 0;
		
		for (int i = 0; i < rowLength(); i++) {			
			for (int j = 0; j < colLength(); j++) {
				v[vIndex++] = m[i][j];
			}
		}
		
		return v;
	}
	
	/**
	 * Column vector.
	 * 
	 * @param cols One-based column index.
	 * @return Column vector.
	 */
	public double[] colVector(int cols) {
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (cols < 1 || cols > colLength()) 
			throw new IllegalArgumentException();
		
		// Get a column vector.
		double[] colVec = new double[rowLength()];
		
		for (int i = 0; i < rowLength(); i++) {
			colVec[i] = m[i][cols - 1];
		}
		
		return colVec;
	}
	
	/**
	 * Row vector.
	 * 
	 * @param rows One-based row index.
	 * @return Row vector.
	 */
	public double[] rowVector(int rows) {
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (rows < 1 || rows > rowLength()) 
			throw new IllegalArgumentException();
		
		// Get a row vector.
		double[] rowVec = new double[colLength()];
		
		for (int i = 0; i < colLength(); i++) {
			rowVec[i] = m[rows - 1][i]; //?
		}
		
		return rowVec;
	}
	
	// Matrix operation.
	/**
	 * Plus operation.
	 * @param om Other matrix.
	 * @return Summation matrix.
	 */
	public Matrix plus(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a summation matrix.
		double[] sv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			sv[i] = v[i] + ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), sv);
	}
	
	/**
	 * Cumulative plus operation.
	 * @param om Other matrix.
	 */
	public void cumulativePlus(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Cumulative plus.
		for (int rows = 1; rows <= rowLength(); rows++) {
			for (int cols = 1; cols <= colLength(); cols++) {
				m[rows - 1][cols - 1] = m[rows - 1][cols - 1] + om.getVal(rows, cols);
			}
		}
	}
	
	/**
	 * Minus operation.
	 * @param om Other matrix.
	 * @return Subtracted matrix.
	 */
	public Matrix minus(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a subtracted matrix.
		double[] sv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			sv[i] = v[i] - ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), sv);
	}
	
	/**
	 * Conduct inner product for two vectors.
	 * @param v1 First vector.
	 * @param v2 Second vector.
	 * @return Inner product value.
	 */
	public static double innerProduct(double[] v1, double[] v2) {
		
		// Check exception.
		// Null.
		if (v1 == null || v2 == null) 
			throw new NullPointerException();
		
		// Parameters.
		if (v1.length < 1 || v2.length < 1 || v1.length != v2.length) 
			throw new IllegalArgumentException();
		
		// Conduct inner product for two vectors.
		double sum = 0.0;
		
		for (int i = 0; i < v1.length; i++) {
			sum += v1[i] * v2[i];
		}
		
		return sum;
	}
	
	/**
	 * Multiply operation.
	 * @param om Operand matrix.
	 * @return Multiplied matrix.
	 */
	public Matrix multiply(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (colLength() != om.rowLength()) {
			throw new IllegalArgumentException();
		}
		
		// Create a multiplied matrix.
		Matrix mm = new Matrix(rowLength(), om.colLength(), 0.0);
		
		for (int i = 0; i < rowLength(); i++) {
			for (int j = 0; j < om.colLength(); j++) {
				mm.setVal(i + 1, j + 1, Matrix.innerProduct(rowVector(i + 1), om.colVector(j + 1)));
			}
		}
		
		return mm;
	}
	
	/**
	 * Transpose operation.
	 * @return Transpose matrix.
	 */
	public Matrix transpose() {
		
		// Check exception.
		checkMatrix();
		
		// Create a transpose matrix.
		Matrix tm = new Matrix(colLength(), rowLength(), 0.0);
		
		for (int i = 0; i < rowLength(); i++) {
			for (int j = 0; j < colLength(); j++) {
				tm.setVal(j + 1, i + 1, m[i][j]);
			}
		}
		
		return tm;
	}
	
	/**
	 * Multiply both matrixes arithmetically.
	 * @param om Operand matrix
	 * @return Arithmetically multiplied matrix.
	 */
	public Matrix arithmeticalMultiply(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create an arithmetically multiplied matrix.
		double[] amv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			amv[i] = v[i] * ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), amv);
	}
	
	/**
	 * Divide this matrix by an operand matrix arithmetically.
	 * @param om Operand matrix
	 * @return Arithmetically divided matrix.
	 */
	public Matrix arithmeticalDivide(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create an arithematically divided matrix.
		double[] adv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			adv[i] = v[i] / ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), adv);
	}
	
	/**
	 * Vertically add.
	 * @param om Operand matrix.
	 * @return Vertically added matrix.
	 */
	public Matrix verticalAdd(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
				
		// Check exception.
		checkMatrix();
				
		// Dimension.
		if (colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a vertically added matrix.
		double[] vav = new double[v.length + ov.length];
		
		for (int i = 0; i < v.length; i++) {
			vav[i] = v[i];
		}
		
		for (int i = v.length; i < v.length + ov.length; i++) {
			vav[i] = ov[i - v.length];
		}
		
		return new Matrix(rowLength() + om.rowLength(), colLength(), vav);
	}

	/**
	 * Horizontally add.
	 * @param om Operand matrix.
	 * @return Horizontally added matrix.
	 */
	public Matrix horizontalAdd(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
				
		// Check exception.
		checkMatrix();
				
		// Dimension.
		if (rowLength() != om.rowLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a horizontally added matrix.
		double[] hav = new double[v.length + ov.length];
		
		for (int i = 0; i < rowLength(); i++) {
			for (int k = 0; k < colLength(); k++) {
				hav[i * (colLength() + om.colLength()) + k] = v[i * colLength() + k];
			}
			
			for (int l = 0; l < om.colLength(); l++) {
				hav[i * (colLength() + om.colLength()) + colLength() + l] 
						= ov[i * om.colLength() + l];
			}
		}
		
		return new Matrix(rowLength(), colLength() + om.colLength(), hav);
	}
	
	/**
	 * Arithmetical power.
	 * @param p Operand matrix.
	 * @return Arithmetically powered matrix.
	 */
	public Matrix arithmeticalPower(double p) {
		
		// Check exception.
		checkMatrix();
		
		// Create a arithmetically powered matrix.
		double[] v = unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = Math.pow(v[i], p);
		}
		
		return new Matrix(rowLength(), colLength(), v);
	}
	
	/**
	 * Matrix + constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix + a constant value. 
	 */
	public static Matrix constArithmeticalPlus(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix + a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] += val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value + matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value + a matrix. 
	 */
	public static Matrix constArithmeticalPlus(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value + a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val + v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Matrix - constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix - a constant value. 
	 */
	public static Matrix constArithmeticalMinus(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix - a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] -= val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value - matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value - a matrix. 
	 */
	public static Matrix constArithmeticalMinus(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value - a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val - v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Matrix * constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix * a constant value. 
	 */
	public static Matrix constArithmeticalMultiply(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix * a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] *= val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value * matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value * a matrix. 
	 */
	public static Matrix constArithmeticalMultiply(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value * a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val * v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Matrix / constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix / a constant value. 
	 */
	public static Matrix constArithmeticalDivide(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix / a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] /= val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value / matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value / a matrix. 
	 */
	public static Matrix constArithmeticalDivide(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value / a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val / v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Get a sub matrix.
	 * @param range row1:row2, col1:col2 -> {row1, row2, col1, col2} array.
	 * @return Sub matrix.
	 */
	public Matrix getSubMatrix(int[] range) {
		
		// Check exception.
		checkMatrix();
		
		// Null.
		if (range == null) 
			throw new NullPointerException();
		
		// Parameter.
		if (range.length < 4 
				| range[0] < 1 | range[0] > rowLength() 
				| range[1] < range[0]
				| range[1] < 1 | range[1] > rowLength()
				| range[2] < 1 | range[2] > colLength() 
				| range[3] < range[2]
				| range[3] < 1 | range[3] > colLength())
			throw new IllegalArgumentException();
		
		// Create a sub matrix.
		Matrix subMatrix = new Matrix(range[1] - range[0] + 1, range[3] - range[2] + 1, 0.0);
		
		for (int i = range[0]; i <= range[1]; i++) {
			for (int j = range[2]; j <= range[3]; j++) {
				subMatrix.setVal(i - range[0] + 1, j - range[2] + 1
						, getVal(i, j));
			}
		}
						
		return subMatrix;
	}
	
	/**
	 * Clone.
	 * @return Copied matrix.
	 */
	public Matrix clone() {
		
		// Check exception.
		checkMatrix();
				
		// Create a clone matrix.
		Matrix cloneMatrix = new Matrix(rowLength(), colLength(), 0.0);
		
		for (int i = 1; i <= rowLength(); i++) {
			for (int j = 1; j <= colLength(); j++) {
				cloneMatrix.setVal(i, j, getVal(i, j));
			}
		}
						
		return cloneMatrix;
	}
	
	/**
	 * Sum.
	 * @return Summed value of all values of this matrix.
	 */
	public double sum() {
		
		// Check exception.
		checkMatrix();
		
		// Sum all values of this matrix.
		double[] v = unrolledVector();
		double sum = 0.0;
		
		for (int i = 0; i < v.length; i++) {
			sum += v[i];
		}
	
		return sum;
	}
	
	/**
	 * Get an identity matrix.
	 * @param d
	 * @return
	 */
	public static Matrix getIdentity(int d) {
		
		// Check exception.
		if (d < 1) 
			throw new IllegalArgumentException();
		
		// Create an identity matrix.
		Matrix im = new Matrix(d, d, 0.0);
		
		for (int i = 1; i <= d ; i++) {
			im.setVal(i, i, 1.0);
		}
		
		return im;
	}
	
	/**
	 *  Calculate a determinant.
	 * @return a determinant value.
	 */
	public double determinant() {
		
		// Check exception.
		checkMatrix();
		
		if (rowLength() != colLength()) 
			throw new IllegalStateException();
		
		// Calculate.
		if (rowLength() == 1) {
			return Math.abs(getVal(1, 1));
		}
		
		// Calculate a determinant value via cofactor expansion along the first row.
		double det = 0.0;
		
		for (int j = 1; j <= colLength(); j++) {
			det += getVal(1, j) * cofactor(this, 1, j);
		}
		
		return det;
	}
	
	/**
	 * Calculate a cofactor value.
	 * @param m
	 * @param row
	 * @param col
	 * @return
	 */
	public static double cofactor(Matrix m, int row, int col) {
		
		// Check exception.
		m.checkMatrix();
		
		if (!((m.rowLength() == m.colLength() && m.rowLength() >= 2))) 
			throw new IllegalArgumentException();
				
		if (!(1 <= row && row <= m.rowLength() 
				&& 1 <= col && col <= m.colLength())) 
			throw new IllegalArgumentException();
		
		// Get a submatrix obtained by deleting a relevant row and column.
		double[] eVals = new double[(m.rowLength() - 1) * (m.colLength() - 1)];
		int count = 0;
		
		// Eliminate values belong to a relevant row and column.
		for (int i = 1; i <= m.rowLength(); i++) {
			if (i == row)
				continue;
			
			for (int j = 1; j <= m.colLength(); j++) {
				if (j == col)
					continue;
				
				eVals[count++] = m.getVal(i, j);
			}
		}
			
		return Math.pow(-1.0, row + col) 
				* minorDeterminant(new Matrix(m.rowLength() - 1, m.colLength() - 1, eVals));
	}
	
	/**
	 *  Calculate a minor determinant.
	 * @return a determinant value.
	 */
	public static double minorDeterminant(Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		if (m.rowLength() != m.colLength()) 
			throw new IllegalStateException();
		
		// Calculate.
		if (m.rowLength() == 1) {
			return Math.abs(m.getVal(1, 1));
		}
		
		// Calculate a determinant value via cofactor expansion along the first row.
		double det = 0.0;
		
		for (int j = 1; j <= m.colLength(); j++) {
			det += m.getVal(1, j) * cofactor(m, 1, j);
		}
		
		return det;
	}
	
	/**
	 * Get an adjoint matrix.
	 * @return
	 */
	public Matrix adjoint() {
		
		// Check exception.
		checkMatrix();
		
		// Create an adjoint matrix.
		Matrix am = new Matrix(rowLength(), colLength(), 0.0);
		
		for (int row = 1; row <= rowLength(); row++) {
			for (int col = 1; col <= colLength(); col++) {
				am.setVal(row, col, cofactor(this, row, col));
			}
		}
		
		return am;
	}
	
	/**
	 * Get an inverse matrix.
	 * @return
	 */
	public Matrix inverse() {
		
		// Check exception.
		checkMatrix();
		
		// Create an inverse matrix.
		return constArithmeticalMultiply(1.0 / determinant(), adjoint());
	}
	
	// Check the matrix state.
	public void checkMatrix() {
		
		// Null.
		if (m == null) 
			throw new IllegalStateException();
	}
}

abstract class AbstractNeuralNetwork implements ICostFunction {

	// Cluster computing flag.
	public final static int CLUSTER_COMPUTING_NONE = 0;
	public final static int CLUSTER_COMPUTING_APACHE_SPARK = 1;
	
	// Accelerating compuing flag.
	public final static int ACCELERATING_COMPUTING_NONE = 0;
	public final static int ACCELERATING_CUDA = 1;
	
	// Classification and regression type.
	public final static int CLASSIFICATION_TYPE = 0;
	public final static int REGRESSION_TYPE = 1;
	
	// Batch mode constant.
	public final static int BATCH_GRADIENT_DESCENT = 0;
	public final static int MINI_BATCH_GRADIENT_DESCENT = 1;
	public final static int STOCHASTIC_GRADIENT_DESCENT = 2;
	
	// Debug flag.
	public boolean DEBUG = true;

	// Constants for the Neural Network model.
	public final static double EPSILON = 0.01;
		
	// Neural network architecture.
	protected int numLayers;
	protected int[] numActs;
	
	// Neural network parameter map.
	public Map<Integer, Matrix> thetas = new HashMap<Integer, Matrix>();
				
	/** Training result. */
	public static class TrainingResult {
		public List<Double> costVals = new ArrayList<Double>();
		public List<Map<Integer, Matrix>> thetaGrads = new ArrayList<Map<Integer, Matrix>>();
		public List<Map<Integer, Matrix>> eThetaGrads = new ArrayList<Map<Integer, Matrix>>();
		public List<Map<Integer, Matrix>> thetaGradsDiffList = new ArrayList<Map<Integer, Matrix>>();
		
		/** Constructor. */
		public TrainingResult() {
		}
	}
	
	// Optimizer.
	protected Optimizer optimizer;
	
	// Classification and regression type.
	protected int classRegType = CLASSIFICATION_TYPE;
	
	// Cluster computing mode.
	protected int clusterComputingMode = CLUSTER_COMPUTING_NONE;
	
	// Accelerating computing mode.
	protected int acceleratingComputingMode = ACCELERATING_COMPUTING_NONE;
	
	/**
	 * Constructor.
	 * @param numLayers Number of layers of NN.
	 * @param numActs Array of the number of activations for each layer.
	 * @param optimizer Optimizer.
	 */
	public AbstractNeuralNetwork(int classRegType
			, int clusterComputingMode
			, int acceleratingComputingMode
			, int numLayers
			, int[] numActs
			, Optimizer optimizer) {
		
		// Check exception.
		// Null.
		if (numActs == null || optimizer == null) 
			throw new NullPointerException();
		
		// Parameters.
		if (classRegType < 0 || classRegType > 1 ||
				clusterComputingMode < 0 || clusterComputingMode > 1 ||
				acceleratingComputingMode < 0 || acceleratingComputingMode > 1 || 
				numLayers < 2 || numActs.length != numLayers)
			throw new IllegalArgumentException();
		
		for (int n : numActs) 
			if (n < 1) 
				throw new IllegalArgumentException();
		
		this.classRegType = classRegType;
		optimizer.classRegType = classRegType;
		this.clusterComputingMode = clusterComputingMode;
		this.acceleratingComputingMode = acceleratingComputingMode;
		
		this.numLayers = numLayers;
		this.numActs = numActs.clone();
		this.optimizer = optimizer;
	}
	
	/**
	 * Set an optimizer.
	 * @param optimizer Optimizer
	 */
	public void setOptimizer(Optimizer optimizer) {
		
		// Check exception.
		// Null.
		if (optimizer == null) 
			throw new NullPointerException();
		
		this.optimizer = optimizer;
	}
	
	/**
	 * Get an optimizer.
	 * @return Optimizer. 
	 */
	public Optimizer getOptimizer() {
		return optimizer;
	}
	
	/**
	 * Train the Neural Network model.
	 * @param X Factor matrix.
	 * @param Y Result matrix.
	 * @return Training result for analysis.
	 */
	public TrainingResult train(Matrix X
			, Matrix Y
			, double lambda
			, int batchMode
			, int numSamplesForMiniBatch
			, int numRepeat
			, int numIter
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio) {

		// Check exception.
		// Null.
		if (X == null || Y == null) 
			throw new NullPointerException();

		// About neural network design.
		boolean result = true;

		// Create and initialize the theta map for each layer.
		createInitThetaMap();
		
		// About batch mode.
		switch (batchMode) {
		case BATCH_GRADIENT_DESCENT: {			
			boolean result1 = (X.rowLength() != (numActs[1 - 1]))
					|| (Y.rowLength() != numActs[numActs.length - 1])
					|| (X.colLength() != Y.colLength());
			
			result = result && result1;
		}
			break;
		case MINI_BATCH_GRADIENT_DESCENT: {
			boolean result1 = (X.rowLength() != (numActs[1 - 1]))
					|| (Y.rowLength() != numActs[numActs.length - 1])
					|| (X.colLength() != Y.colLength());
						
			boolean result2 = (numIter * numSamplesForMiniBatch <= X.colLength());
			
			result = result && result1 && result2;	
		}
			break;
		case STOCHASTIC_GRADIENT_DESCENT: {			
			boolean result1 = (X.rowLength() != (numActs[1 - 1]))
					|| (Y.rowLength() != numActs[numActs.length - 1])
					|| (X.colLength() != Y.colLength());
						
			boolean result2 = (numIter <= X.colLength());
			
			result = result && result1 && result2;	
		}
			break;
		}
		
		if (result)
			throw new IllegalArgumentException();
		
		// Calculate parameters to minimize the Neural Network cost function.
		return calParams(X
				, Y
				, lambda
				, batchMode
				, numSamplesForMiniBatch
				, numRepeat
				, numIter
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio);
	}
		
	/**
	 * Feedforward for classification.
	 * @param X Matrix of input values.
	 * @return Feedforward probability matrix.
	 */
	protected Matrix feedForwardC(Matrix X) {
		@SuppressWarnings("unused")
		int numSamples = X.colLength();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();

		// The first activation.
		actMatrixes.put(1, X);
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get activation matrixes.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(2, secondActMatrix);
		
		// From the second activation.
		for (int i = 3; i <= numLayers; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i - 1).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers); 
		
		return finalActMatrix;
	}
	
	/**
	 * Feedforward for regression.
	 * @param X Matrix of input values.
	 * @return Feedforward probability matrix.
	 */
	protected Matrix feedForwardR(Matrix X) {
		@SuppressWarnings("unused")
		int numSamples = X.colLength();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();

		// The first activation.
		actMatrixes.put(1, X);
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get activation matrixes.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(2, secondActMatrix);
		
		// From the second activation.
		for (int i = 3; i <= numLayers - 1; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i - 1).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// To the final activation for regression.
		// Get a bias term added activation.
		Matrix bias = new Matrix(1, actMatrixes.get(numLayers - 1).colLength(), 1.0);
		Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(numLayers - 1));
		
		// Get an activation matrix.
		Matrix actMatrix = thetas.get(numLayers - 1).multiply(biasAddedActMatrix);
		actMatrixes.put(numLayers, actMatrix);
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers); 
		
		return finalActMatrix;
	}
	
	// Calculate parameters to minimize the Neural Network cost function.
	private TrainingResult calParams(Matrix X
			, Matrix Y
			, double lambda
			, int batchMode
			, int numSamplesForMiniBatch
			, int numRepeat
			, int numIter
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio) {
		
		// Conduct an optimization method to get optimized theta values.
		TrainingResult tResult = new TrainingResult();
		
		// Optmization'a parameters are assumed to be valid. ??
		switch (batchMode) {
		case BATCH_GRADIENT_DESCENT: {
			if (optimizer instanceof NonlinearCGOptimizer) {
				List<CostFunctionResult> rs = new ArrayList<CostFunctionResult>();
				
				thetas = ((NonlinearCGOptimizer) optimizer).fmincg(this
						, clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, thetas // ?
						, numIter
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio
						, rs);
				
				// Calculate training results.
				for (CostFunctionResult r : rs) {
					
					// Add a cost value.
					tResult.costVals.add(r.J);
					
					// Calculate difference between calculated and estimated gradient descent values
					if (isGradientChecking) {
						Map<Integer, Matrix> diffThetas = new HashMap<Integer, Matrix>();
						
						for (int j = 1; j <= numLayers - 1; j++) {
							Matrix diff = r.thetaGrads.get(j).minus(r.eThetaGrads.get(j));
							diffThetas.put(j, diff);
						}
											
						tResult.thetaGrads.add(r.thetaGrads);
						tResult.eThetaGrads.add(r.eThetaGrads);
						tResult.thetaGradsDiffList.add(diffThetas);
					}
				}
			} else if (optimizer instanceof LBFGSOptimizer) {
				List<CostFunctionResult> rs = new ArrayList<CostFunctionResult>();
				
				thetas = ((LBFGSOptimizer) optimizer).minimize(this
						, clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, thetas // ?
						, numIter
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio
						, rs);
				
				// Calculate training results.
				for (CostFunctionResult r : rs) {
					
					// Add a cost value.
					tResult.costVals.add(r.J);
					
					// Calculate difference between calculated and estimated gradient descent values
					if (isGradientChecking) {
						Map<Integer, Matrix> diffThetas = new HashMap<Integer, Matrix>();
						
						for (int j = 1; j <= numLayers - 1; j++) {
							Matrix diff = r.thetaGrads.get(j).minus(r.eThetaGrads.get(j));
							diffThetas.put(j, diff);
						}
											
						tResult.thetaGrads.add(r.thetaGrads);
						tResult.eThetaGrads.add(r.eThetaGrads);
						tResult.thetaGradsDiffList.add(diffThetas);
					}
				}
			}
		}
			break;
		case MINI_BATCH_GRADIENT_DESCENT: {			
		}
			break;
		case STOCHASTIC_GRADIENT_DESCENT: {						
		}
			break;
		}
		
		return tResult;
	}

	// Shuffle samples randomly.
	private void shuffleSamples(Matrix X, Matrix Y) {
		
		// Input matrixes are assumed to be valid.
		Matrix SX = X.clone();
		Matrix SY = Y.clone();
		
		Random rnd = new Random();
		int count = 0;
		int rCount = 0;
		int bound = X.colLength();
		List<Integer> indexes = new ArrayList<Integer>();
		
		// Shuffle.
		do {
			int index = rnd.nextInt(bound) + 1;
			
			if (!indexes.contains(index)) {
				for (int rows = 1; rows <= X.rowLength(); rows++) {
					SX.setVal(rows, index, X.getVal(rows, index));
				}
				
				for (int rows = 1; rows <= Y.rowLength(); rows++) {
					SY.setVal(rows, index, Y.getVal(rows, index));
				}
				
				indexes.add(index);
				count++;
			}
			
			rCount++;
			
		} while (count < X.colLength());
		
		X = SX;
		Y = SY;
	}
	
	// Calculate the Neural Network cost function and theta gradient for classification.
	public CostFunctionResult costFunctionC(int clusterComputingMode
			, int acceratingComputingMode
			, Matrix X
			, Matrix Y
			, final Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio) {
		
		// Calculate the Neural Network cost function.
		final int numSamples = X.colLength();
		final int iNumSamples = (int)(numSamples * JEstimationRatio) < 1 
				? 1 : (int)(numSamples * JEstimationRatio); 
		CostFunctionResult result = new CostFunctionResult();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();

		// The first activation.
		actMatrixes.put(1, X);
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get activation matrixes.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(2, secondActMatrix);
		
		// From the second activation.
		for (int i = 3; i <= numLayers; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i - 1).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers); 
		
		// Estimated a cost matrix (m x m) via sampling.
		Matrix cost = null;
		
		if (JEstimationFlag) {

			// Get 100 samples randomly.
			Random rnd = new Random();
			int count = 0;
			int bound = X.colLength();
					
			List<Integer> indexes = new ArrayList<Integer>();
			
			do {
				int index = rnd.nextInt(bound) + 1;
				
				if (!indexes.contains(index)) {					
					indexes.add(index);
					count++;
				}				
			} while (count < iNumSamples);
			
			Matrix iFinalActMatrix = finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(0), indexes.get(0)});
			Matrix iY = Y.getSubMatrix(
					new int[] {1, Y.rowLength(), indexes.get(0), indexes.get(0)});
			
			for (int l = 1; l < iNumSamples; l++) {
				iFinalActMatrix = iFinalActMatrix.horizontalAdd(finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(l), indexes.get(l)}));
				iY = iY.horizontalAdd(Y.getSubMatrix(
						new int[] {1, Y.rowLength(), indexes.get(l), indexes.get(l)}));
			}
			
			Matrix term1 = Matrix.constArithmeticalMultiply(-1.0, iY.transpose()).multiply(log(iFinalActMatrix));
			Matrix term2 = Matrix.constArithmeticalMinus(1.0, iY).transpose()
					.multiply(log(Matrix.constArithmeticalMinus(1.0, iFinalActMatrix)));
			Matrix term3 = term1.minus(term2);
			cost = Matrix.constArithmeticalDivide(term3, iNumSamples);
		} else {
			Matrix term1 = Matrix.constArithmeticalMultiply(-1.0, Y.transpose()).multiply(log(finalActMatrix));
			Matrix term2 = Matrix.constArithmeticalMinus(1.0, Y).transpose()
					.multiply(log(Matrix.constArithmeticalMinus(1.0, finalActMatrix)));
			Matrix term3 = term1.minus(term2);
			cost = Matrix.constArithmeticalDivide(term3, numSamples);
		}

		// Cost function.
		double J = 0.0;
		
		if (JEstimationFlag) {
			for (int i = 1; i <= iNumSamples; i++) {
				J += cost.getVal(i, i);
			}
		} else {
			for (int i = 1; i <= numSamples; i++) {
				J += cost.getVal(i, i);
			}
		}

		// Regularized cost function.
		double sum = 0.0;
		
		for (int key : thetas.keySet()) {
			Matrix theta = thetas.get(key);
			int[] range = {1, theta.rowLength(), 2, theta.colLength()};
			
			sum += theta.getSubMatrix(range).arithmeticalPower(2.0).sum();
		}
		
		J += lambda / (2.0 * numSamples) * sum;
	
		result.J = J;
		
		// Estimate gradient descent values for gradient checking.
		if (isGradientChecking) {
			for (int i = 1; i <= numLayers - 1; i++) {
				Matrix eThetaGrad = new Matrix(numActs[i], numActs[i - 1] + 1, 0.0);
				
				// Estimate theta gradient values.
				for (int rows = 1; rows <= eThetaGrad.rowLength(); rows++) {
					for (int cols = 1; cols <= eThetaGrad.colLength(); cols++) {
						
						// Calculate the cost value for theta + epsilon.
						Map<Integer, Matrix> ePlusThetas = new HashMap<Integer, Matrix>();
						
						for (int v : thetas.keySet()) {
							ePlusThetas.put(v, thetas.get(v).clone());
						}
						
						ePlusThetas.get(i).setVal(rows, cols, ePlusThetas.get(i).getVal(rows, cols) + EPSILON);
						double ePlusCostVal = calCostValForThetasC(X, Y, ePlusThetas, lambda
								, JEstimationFlag, JEstimationRatio);
						
						// Calculate the cost value for theta - epsilon.
						Map<Integer, Matrix> eMinusThetas = new HashMap<Integer, Matrix>();
						
						for (int v : thetas.keySet()) {
							eMinusThetas.put(v, thetas.get(v).clone());
						}
						
						eMinusThetas.get(i).setVal(rows, cols, eMinusThetas.get(i).getVal(rows, cols) - EPSILON);
						double eMinusCostVal = calCostValForThetasC(X, Y, eMinusThetas, lambda
								, JEstimationFlag, JEstimationRatio);
						
						eThetaGrad.setVal(rows, cols, (ePlusCostVal - eMinusCostVal) / ( 2.0 * EPSILON));
					}
				}
				
				result.eThetaGrads.put(i, eThetaGrad);
			}
		}

		// Backpropagation.
		if (clusterComputingMode == CLUSTER_COMPUTING_NONE) {
			
			// Calculate character deltas for each layer.
			Map<Integer, Matrix> cTotalSumDeltas = new HashMap<Integer, Matrix>();
			
			// Initialize character deltas.
			for (int i = 1; i <= numLayers - 1; i++) {
				cTotalSumDeltas.put(i, new Matrix(numActs[i], numActs[i - 1] + 1, 0.0));
			}
			
			for (int i = 1; i <= numSamples; i++) {
				int[] range1 = {1, firstBiasAddedActMatrix.rowLength(), i, i};
				Matrix a1 = firstBiasAddedActMatrix.getSubMatrix(range1);
				a1.index = i;
				
				// Feedforward.
				Map<Integer, Matrix> zs = new HashMap<Integer, Matrix>();
				Map<Integer, Matrix> as = new HashMap<Integer, Matrix>();
				as.put(1, a1);
				
				for (int k = 2; k <= numLayers - 1; k++) {
					Matrix z = thetas.get(k - 1).multiply(as.get(k - 1));
					Matrix a =  new Matrix(1, 1, 1.0).verticalAdd(sigmoid(z));
					
					zs.put(k, z);
					as.put(k, a);
				}
				
				Matrix z = thetas.get(numLayers - 1).multiply(as.get(numLayers - 1));
				Matrix a = sigmoid(z);
				
				zs.put(numLayers, z);
				as.put(numLayers, a);
				
				// Calculate delta vectors for each layer.
				Map<Integer, Matrix> deltas = new HashMap<Integer, Matrix>();
				int[] range = {1, Y.rowLength(), i, i};
				
				deltas.put(numLayers, as.get(numLayers).minus(Y.getSubMatrix(range)));
				
				for (int k = numLayers - 1; k >= 2; k--) {					
					Matrix biasAddedSG = new Matrix(1, 1, 1.0).verticalAdd(sigmoidGradient(zs.get(k)));
					Matrix preDelta = thetas.get(k).transpose().multiply(deltas.get(k + 1))
							.arithmeticalMultiply(biasAddedSG);
					
					int[] iRange = {2, preDelta.rowLength(), 1, 1};					
					Matrix delta = preDelta.getSubMatrix(iRange); 
					
					deltas.put(k, delta);	
				}
				
				// Accumulate the gradient.
				for (int k = numLayers - 1; k >= 1; k--) {
					cTotalSumDeltas.get(k).cumulativePlus(deltas.get(k + 1).multiply(as.get(k).transpose()));
				}
			}
							
			// Obtain the regularized gradient.		
			for (int i = 1; i <= numLayers - 1; i++) {
				int[] range = {1, cTotalSumDeltas.get(i).rowLength(), 2, cTotalSumDeltas.get(i).colLength()};
				Matrix preThetaGrad = Matrix.constArithmeticalDivide(cTotalSumDeltas.get(i).getSubMatrix(range), (double)numSamples)
						.plus(Matrix.constArithmeticalMultiply(lambda / numSamples
								, thetas.get(i).getSubMatrix(range)));
				
				int[] range2 = {1, cTotalSumDeltas.get(i).rowLength(), 1, 1};  
				Matrix thetaGrad = Matrix.constArithmeticalDivide(cTotalSumDeltas.get(i).getSubMatrix(range2), (double)numSamples)
						.horizontalAdd(preThetaGrad);
				
				result.thetaGrads.put(i, thetaGrad);
			}
		} else if (clusterComputingMode == CLUSTER_COMPUTING_APACHE_SPARK) {
		}
		
		return result;
	}

	// Calculate the Neural Network cost function and theta gradient for regression.
	public CostFunctionResult costFunctionR(int clusterComputingMode
			, int acceratingComputingMode
			, Matrix X
			, Matrix Y
			, final Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio) {
		
		// Calculate the Neural Network cost function.
		final int numSamples = X.colLength();
		final int iNumSamples = (int)(numSamples * JEstimationRatio) < 1 ? 1 : (int)(numSamples * JEstimationRatio); 
		CostFunctionResult result = new CostFunctionResult();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();

		// The first activation.
		actMatrixes.put(1, X);
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get activation matrixes.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(2, secondActMatrix);
		
		// From the second activation.
		for (int i = 3; i < numLayers; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i - 1).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// To the final activation for regression.
		// Get a bias term added activation.
		Matrix bias = new Matrix(1, actMatrixes.get(numLayers - 1).colLength(), 1.0);
		Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(numLayers - 1));
		
		// Get an activation matrix.
		Matrix actMatrix = thetas.get(numLayers - 1).multiply(biasAddedActMatrix);
		actMatrixes.put(numLayers, actMatrix);
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers); 
		
		// Estimated a cost matrix (m x m) via sampling.
		Matrix cost = null;
		
		if (JEstimationFlag) {

			// Get 100 samples randomly.
			Random rnd = new Random();
			int count = 0;
			int bound = X.colLength();
					
			List<Integer> indexes = new ArrayList<Integer>();
			
			do {
				int index = rnd.nextInt(bound) + 1;
				
				if (!indexes.contains(index)) {					
					indexes.add(index);
					count++;
				}				
			} while (count < iNumSamples);
			
			Matrix iFinalActMatrix = finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(0), indexes.get(0)});
			Matrix iY = Y.getSubMatrix(
					new int[] {1, Y.rowLength(), indexes.get(0), indexes.get(0)});
			
			for (int l = 1; l < iNumSamples; l++) {
				iFinalActMatrix = iFinalActMatrix.horizontalAdd(finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(l), indexes.get(l)}));
				iY = iY.horizontalAdd(Y.getSubMatrix(
						new int[] {1, Y.rowLength(), indexes.get(l), indexes.get(l)}));
			}
			
			Matrix term1 = iY.minus(iFinalActMatrix).arithmeticalPower(2.0);
			cost = Matrix.constArithmeticalDivide(term1, iNumSamples);
		} else {
			Matrix term1 = Y.minus(finalActMatrix).arithmeticalPower(2.0);
			cost = Matrix.constArithmeticalDivide(term1, numSamples);
		}

		// Cost function.
		double J = 0.0;
		
		if (JEstimationFlag) {
			for (int i = 1; i <= iNumSamples; i++) {
				for (int j = 1; j <= Y.rowLength(); j++) {
					J += cost.getVal(j, i);
				}
			}
		} else {
			for (int i = 1; i <= numSamples; i++) {
				for (int j = 1; j <= Y.rowLength(); j++) {
					J += cost.getVal(j, i);
				}
			}
		}

		// Regularized cost function.
		double sum = 0.0;
		
		for (int key : thetas.keySet()) {
			Matrix theta = thetas.get(key);
			int[] range = {1, theta.rowLength(), 2, theta.colLength()};
			
			sum += theta.getSubMatrix(range).arithmeticalPower(2.0).sum();
		}
		
		J += lambda / (2.0 * numSamples) * sum; // iNumSamples?
	
		result.J = J;
		
		// Estimate gradient descent values for gradient checking.
		if (isGradientChecking) {
			for (int i = 1; i <= numLayers - 1; i++) {
				Matrix eThetaGrad = new Matrix(numActs[i], numActs[i - 1] + 1, 0.0);
				
				// Estimate theta gradient values.
				for (int rows = 1; rows <= eThetaGrad.rowLength(); rows++) {
					for (int cols = 1; cols <= eThetaGrad.colLength(); cols++) {
						
						// Calculate the cost value for theta + epsilon.
						Map<Integer, Matrix> ePlusThetas = new HashMap<Integer, Matrix>();
						
						for (int v : thetas.keySet()) {
							ePlusThetas.put(v, thetas.get(v).clone());
						}
						
						ePlusThetas.get(i).setVal(rows, cols, ePlusThetas.get(i).getVal(rows, cols) + EPSILON);
						double ePlusCostVal = calCostValForThetasR(X, Y, ePlusThetas, lambda
								, JEstimationFlag, JEstimationRatio);
						
						// Calculate the cost value for theta - epsilon.
						Map<Integer, Matrix> eMinusThetas = new HashMap<Integer, Matrix>();
						
						for (int v : thetas.keySet()) {
							eMinusThetas.put(v, thetas.get(v).clone());
						}
						
						eMinusThetas.get(i).setVal(rows, cols, eMinusThetas.get(i).getVal(rows, cols) - EPSILON);
						double eMinusCostVal = calCostValForThetasR(X, Y, eMinusThetas, lambda
								, JEstimationFlag, JEstimationRatio);
						
						eThetaGrad.setVal(rows, cols, (ePlusCostVal - eMinusCostVal) / ( 2.0 * EPSILON));
					}
				}
				
				result.eThetaGrads.put(i, eThetaGrad);
			}
		}

		// Backpropagation.
		if (clusterComputingMode == CLUSTER_COMPUTING_NONE) {
			
			// Calculate character deltas for each layer.
			Map<Integer, Matrix> cTotalSumDeltas = new HashMap<Integer, Matrix>();
			
			// Initialize character deltas.
			for (int i = 1; i <= numLayers - 1; i++) {
				cTotalSumDeltas.put(i, new Matrix(numActs[i], numActs[i - 1] + 1, 0.0));
			}
			
			for (int i = 1; i <= numSamples; i++) {
				int[] range1 = {1, firstBiasAddedActMatrix.rowLength(), i, i};
				Matrix a1 = firstBiasAddedActMatrix.getSubMatrix(range1);
				a1.index = i;
				
				// Feedforward.
				Map<Integer, Matrix> zs = new HashMap<Integer, Matrix>();
				Map<Integer, Matrix> as = new HashMap<Integer, Matrix>();
				as.put(1, a1);
				
				for (int k = 2; k <= numLayers - 1; k++) {
					Matrix z = thetas.get(k - 1).multiply(as.get(k - 1));
					Matrix a =  new Matrix(1, 1, 1.0).verticalAdd(sigmoid(z));
					
					zs.put(k, z);
					as.put(k, a);
				}
				
				Matrix z = thetas.get(numLayers - 1).multiply(as.get(numLayers - 1));
				Matrix a = z; // Regression.
				
				zs.put(numLayers, z);
				as.put(numLayers, a);
				
				// Calculate delta vectors for each layer.
				Map<Integer, Matrix> deltas = new HashMap<Integer, Matrix>();
				int[] range = {1, Y.rowLength(), i, i};
				
				deltas.put(numLayers, Matrix.constArithmeticalMultiply(2.0
						, as.get(numLayers).minus(Y.getSubMatrix(range))));
				
				for (int k = numLayers - 1; k >= 2; k--) {					
					Matrix biasAddedSG = new Matrix(1, 1, 1.0).verticalAdd(sigmoidGradient(zs.get(k)));
					Matrix preDelta = thetas.get(k).transpose().multiply(deltas.get(k + 1))
							.arithmeticalMultiply(biasAddedSG);
					
					int[] iRange = {2, preDelta.rowLength(), 1, 1};					
					Matrix delta = preDelta.getSubMatrix(iRange); 
					
					deltas.put(k, delta);	
				}
				
				// Accumulate the gradient.
				for (int k = numLayers - 1; k >= 1; k--) {
					cTotalSumDeltas.get(k).cumulativePlus(deltas.get(k + 1).multiply(as.get(k).transpose()));
				}
			}
							
			// Obtain the regularized gradient.		
			for (int i = 1; i <= numLayers - 1; i++) {
				int[] range = {1, cTotalSumDeltas.get(i).rowLength(), 2, cTotalSumDeltas.get(i).colLength()};
				Matrix preThetaGrad = Matrix.constArithmeticalDivide(cTotalSumDeltas.get(i).getSubMatrix(range), (double)numSamples)
						.plus(Matrix.constArithmeticalMultiply(lambda / numSamples
								, thetas.get(i).getSubMatrix(range)));
				
				int[] range2 = {1, cTotalSumDeltas.get(i).rowLength(), 1, 1};  
				Matrix thetaGrad = Matrix.constArithmeticalDivide(cTotalSumDeltas.get(i).getSubMatrix(range2), (double)numSamples)
						.horizontalAdd(preThetaGrad);
				
				result.thetaGrads.put(i, thetaGrad);
			}
		} else if (clusterComputingMode == CLUSTER_COMPUTING_APACHE_SPARK) {
		}
		
		return result;
	}
		
	// Calculate a cost value for thetas.
	private double calCostValForThetasC(Matrix X, Matrix Y, Map<Integer, Matrix> thetas
			, double lambda, boolean JEstimationFlag, double JEstimationRatio) {
		
		// Calculate the Neural Network cost function.
		final int numSamples = X.colLength();
		final int iNumSamples = (int)(numSamples * JEstimationRatio) < 1 ? 1 : (int)(numSamples * JEstimationRatio); 
		CostFunctionResult result = new CostFunctionResult();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();

		// The first activation.
		actMatrixes.put(1, X);
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get activation matrixes.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(2, secondActMatrix);
		
		// From the second activation.
		for (int i = 3; i <= numLayers; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i - 1).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers); 
		
		// Estimated a cost matrix (m x m) via sampling.
		Matrix cost = null;
		
		if (JEstimationFlag) {

			// Get 100 samples randomly.
			Random rnd = new Random();
			int count = 0;
			int bound = X.colLength();
			List<Integer> indexes = new ArrayList<Integer>();
			
			do {
				int index = rnd.nextInt(bound) + 1;
				
				if (!indexes.contains(index)) {					
					indexes.add(index);
					count++;
				}				
			} while (count < iNumSamples);
			
			Matrix iFinalActMatrix = finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(0), indexes.get(0)});
			Matrix iY = Y.getSubMatrix(
					new int[] {1, Y.rowLength(), indexes.get(0), indexes.get(0)});
			
			for (int l = 1; l < iNumSamples; l++) {
				iFinalActMatrix = iFinalActMatrix.horizontalAdd(finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(l), indexes.get(l)}));
				iY = iY.horizontalAdd(Y.getSubMatrix(
						new int[] {1, Y.rowLength(), indexes.get(l), indexes.get(l)}));
			}
			
			Matrix term1 = Matrix.constArithmeticalMultiply(-1.0, iY.transpose()).multiply(log(iFinalActMatrix));
			Matrix term2 = Matrix.constArithmeticalMinus(1.0, iY).transpose()
					.multiply(log(Matrix.constArithmeticalMinus(1.0, iFinalActMatrix)));
			Matrix term3 = term1.minus(term2);
			cost = Matrix.constArithmeticalDivide(term3, iNumSamples);
		} else {
			Matrix term1 = Matrix.constArithmeticalMultiply(-1.0, Y.transpose()).multiply(log(finalActMatrix));
			Matrix term2 = Matrix.constArithmeticalMinus(1.0, Y).transpose()
					.multiply(log(Matrix.constArithmeticalMinus(1.0, finalActMatrix)));
			Matrix term3 = term1.minus(term2);
			cost = Matrix.constArithmeticalDivide(term3, numSamples);
		}

		// Cost function.
		double J = 0.0;
		
		if (JEstimationFlag) {
			for (int i = 1; i <= iNumSamples; i++) {
				J += cost.getVal(i, i);
			}
		} else {
			for (int i = 1; i <= numSamples; i++) {
				J += cost.getVal(i, i);
			}
		}
		
		// Regularized cost function.
		double sum = 0.0;
		
		for (int key : thetas.keySet()) {
			Matrix theta = thetas.get(key);
			int[] range = {1, theta.rowLength(), 2, theta.colLength()};
			
			sum += theta.getSubMatrix(range).arithmeticalPower(2.0).sum();
		}
		
		J += lambda / (2.0 * numSamples) * sum;
			
		return J;
	}

	// Calculate a cost value for thetas.
	private double calCostValForThetasR(Matrix X, Matrix Y, Map<Integer, Matrix> thetas
			, double lambda, boolean JEstimationFlag, double JEstimationRatio) {
		
		// Calculate the Neural Network cost function.
		final int numSamples = X.colLength();
		final int iNumSamples = (int)(numSamples * JEstimationRatio) < 1 ? 1 : (int)(numSamples * JEstimationRatio); 
		CostFunctionResult result = new CostFunctionResult();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();

		// The first activation.
		actMatrixes.put(1, X);
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get activation matrixes.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(2, secondActMatrix);
		
		// From the second activation.
		for (int i = 3; i <= numLayers - 1; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i - 1).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// To the final activation for regression.
		// Get a bias term added activation.
		Matrix bias = new Matrix(1, actMatrixes.get(numLayers - 1).colLength(), 1.0);
		Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(numLayers - 1));
		
		// Get an activation matrix.
		Matrix actMatrix = thetas.get(numLayers - 1).multiply(biasAddedActMatrix);
		actMatrixes.put(numLayers, actMatrix);
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers); 
		
		// Estimated a cost matrix (m x m) via sampling.
		Matrix cost = null;
		
		if (JEstimationFlag) {

			// Get 100 samples randomly.
			Random rnd = new Random();
			int count = 0;
			int bound = X.colLength();
			List<Integer> indexes = new ArrayList<Integer>();
			
			do {
				int index = rnd.nextInt(bound) + 1;
				
				if (!indexes.contains(index)) {					
					indexes.add(index);
					count++;
				}				
			} while (count < iNumSamples);
			
			Matrix iFinalActMatrix = finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(0), indexes.get(0)});
			Matrix iY = Y.getSubMatrix(
					new int[] {1, Y.rowLength(), indexes.get(0), indexes.get(0)});
			
			for (int l = 1; l < iNumSamples; l++) {
				iFinalActMatrix = iFinalActMatrix.horizontalAdd(finalActMatrix.getSubMatrix(
					new int[] {1, finalActMatrix.rowLength(), indexes.get(l), indexes.get(l)}));
				iY = iY.horizontalAdd(Y.getSubMatrix(
						new int[] {1, Y.rowLength(), indexes.get(l), indexes.get(l)}));
			}
			
			Matrix term1 = iY.minus(iFinalActMatrix).arithmeticalPower(2.0);
			cost = Matrix.constArithmeticalDivide(term1, iNumSamples);
		} else {
			Matrix term1 = Y.minus(finalActMatrix).arithmeticalPower(2.0);
			cost = Matrix.constArithmeticalDivide(term1, numSamples);
		}

		// Cost function.
		double J = 0.0;
		
		if (JEstimationFlag) {
			for (int i = 1; i <= iNumSamples; i++) {
				for (int j = 1; j <= Y.rowLength(); j++) {
					J += cost.getVal(j, i);
				}
			}
		} else {
			for (int i = 1; i <= numSamples; i++) {
				for (int j = 1; j <= Y.rowLength(); j++) {
					J += cost.getVal(j, i);
				}
			}
		}
		
		// Regularized cost function.
		double sum = 0.0;
		
		for (int key : thetas.keySet()) {
			Matrix theta = thetas.get(key);
			int[] range = {1, theta.rowLength(), 2, theta.colLength()};
			
			sum += theta.getSubMatrix(range).arithmeticalPower(2.0).sum();
		}
		
		J += lambda / (2.0 * numSamples) * sum; //?
			
		return J;
	}	
	// Exponential log.
	private Matrix log(Matrix m) {
		
		// Create a matrix having log function values for the input matrix.
		Matrix lm = new Matrix(m.rowLength(), m.colLength(), 0.0);
		
		for (int rows = 1; rows <= m.rowLength(); rows++) {
			for (int cols = 1; cols <= m.colLength(); cols++) {
				lm.setVal(rows, cols
						, Math.log(m.getVal(rows, cols)));
			}
		}
		
		return lm;
	}
	
	// Sigmoid function.
	private static Matrix sigmoid(Matrix m) {
		
		// Create a matrix having sigmoid function values for the input matrix.
		Matrix sm = new Matrix(m.rowLength(), m.colLength(), 0.0);
		
		for (int rows = 1; rows <= m.rowLength(); rows++) {
			for (int cols = 1; cols <= m.colLength(); cols++) {
				sm.setVal(rows, cols
						, 1.0 /(1.0 + Math.pow(Math.E, -1.0 * m.getVal(rows, cols))));
			}
		}
		
		return sm;
	}
	
	// Sigmoid gradient function.
	private static Matrix sigmoidGradient(Matrix m) {
		
		// Create a sigmoid gradient matrix.
		Matrix one = new Matrix(m.rowLength(), m.colLength(), 1.0);
		Matrix sdm = sigmoid(m).arithmeticalMultiply(one.minus(sigmoid(m)));
		
		return sdm;
	}
	
	// Create and initialize the theta map for each layer.
	private void createInitThetaMap() {
		Random rand = new Random(); //?

		thetas.clear();
		
		for (int i = 1; i <= numLayers - 1; i++) {
			
			// Create a theta.
			Matrix theta = new Matrix(numActs[i], numActs[i - 1] + 1, 0.0); 

			// Calculate e.
			double numInLayerAct = numActs[i - 1];
			double numOutLayerAct = numActs[i];
			
			double e = Math.sqrt(6.0) / Math.sqrt(numInLayerAct + numOutLayerAct);
						
			// Initialize the created theta.
			int count = 0;
			for (int rows = 1; rows <= theta.rowLength(); rows++ ) {
				for (int cols = 1; cols <= theta.colLength(); cols++) {
					theta.setVal(rows, cols, 0.0); //rand.nextDouble() * 2.0 * e - e);
				}
			}
			
			thetas.put(i, theta);	
		}
	}
	
	// Print a message.
	public static void printMsg(String msg) {
		System.out.println(msg);
	}

	// Print a message at a same line.
	public static void printMsgatSameLine(String msg) {
		System.out.print("\r" + msg);
	}
}

/**
 * Mutiple Myeloma Progress Predictor.
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 26, 2016
 */
public class MMRF {
	
	/** Debug flag. */
	public final static boolean DEBUG = true; 
	
	/** Constants. */
	public final static int TOTAL_NUM_GENES = 18898;
	public final static double Lambda = 0.00;
	
	public final static int TRAINING_MODE_STATIC = 0;
	public final static int TRAINING_MODE_DYNAMIC = 1;
	
	
	/** MMRF info. */
	public class MMRFInfo {
		public Matrix expr_avg;
		public Matrix expr_diff;
		public Matrix mutation;
		public Matrix prog_obs_time;
		public Matrix survival_risk;
	}
	
	/** MM progress predictor. */
	public class MMProgressPredictor {
		
		// Constants for the neural network regression.
		public final static int clusterComputingMode = 0;
		public final static int acceleratingComputingMode = 0;
		
		// Multiple Myeloma progress prediction model using hierarchical NN model layers.
		// Superficial NN model layer.
		// Partial neural network regression models for gene expression and survival risk.
		public List<NeuralNetworkRegression> partialExpRiskNNs = 		
				new ArrayList<NeuralNetworkRegression>();
		
		// Partial neural network regression models for gene variation and survival risk.
		public List<NeuralNetworkRegression> partialVarRiskNNs = 		
				new ArrayList<NeuralNetworkRegression>();
		
		// Partial neural network regression models for gene mutation and survival risk.
		public List<NeuralNetworkRegression> partialMutRiskNNs = 		
				new ArrayList<NeuralNetworkRegression>();
		
		// Local NN model layer considering local gene expression, variation and mutation dependencies.
		public List<NeuralNetworkRegression> localNNs = 
				new ArrayList<NeuralNetworkRegression>();
		
		// Final abstract NN model layer.
		public NeuralNetworkRegression abstractNN = null;
		
		// Model configuration parameters.
		private int partialNumGenes;
		private int localLayersDepth;
		private int abstractLayerDepth;
		private int numSuperficialNNModels;
		
		// Number of iteration.
		public int numIter = 20;
		
		// NN parameters base64 encoded string for static training.
		public String nnParamsBase64Str ="";
		
		/**
		 * Constructor.
		 * 
		 * @param localLayerDepth
		 * @param abstractLayerDepth
		 */
		public MMProgressPredictor(int localLayerDepth
				, int abstractLayerDepth) {
			
			// Input parameters are assumed to be valid.
			
			partialNumGenes = numSymNeighborGenes * 2 + 1;
			this.localLayersDepth = localLayerDepth;
			this.abstractLayerDepth = abstractLayerDepth;
			
			// Configure the model.
			configureModel(localLayerDepth, abstractLayerDepth);
		}
		
		// Configure the model.
		private void configureModel(int localLayerDepth
				, int abstractLayerDepth) {
			
			// Initialize each NN models.
			partialExpRiskNNs.clear();
			partialVarRiskNNs.clear();
			partialMutRiskNNs.clear();
			localNNs.clear();
			abstractNN = null;
			
			// Configure the superficial NN model layer.
			// Calculate the number of NN models.
			numSuperficialNNModels = pivotGenes.length;
			
			// Create NN models.
			// [NUM_GENES, 2000, 2000, 200, 200, 100, 1]
			int numLayers = 7;
			int[] numActs = {partialNumGenes
					, Math.max(10, partialNumGenes / 2)
					, Math.max(10, partialNumGenes / 2)
					, Math.max(5, partialNumGenes / 4)
					, Math.max(5, partialNumGenes / 4)
					, Math.max(3, partialNumGenes / 6)
					, 1}; 

			for (int i = 0; i < numSuperficialNNModels; i++) {
				partialExpRiskNNs.add(new NeuralNetworkRegression(clusterComputingMode
						, acceleratingComputingMode
						, numLayers
						, numActs
						, new NonlinearCGOptimizer()));
				partialVarRiskNNs.add(new NeuralNetworkRegression(clusterComputingMode
						, acceleratingComputingMode
						, numLayers
						, numActs
						, new NonlinearCGOptimizer()));
				partialMutRiskNNs.add(new NeuralNetworkRegression(clusterComputingMode
						, acceleratingComputingMode
						, numLayers
						, numActs
						, new NonlinearCGOptimizer()));
			}
			
			// Configure the local NN model layer.
			int numLayersLocal = localLayerDepth;
			int[] numActsLocal = new int[numLayersLocal];
			
			numActsLocal[0] = 3;
			
			for (int i = 1; i < numLayersLocal - 1; i++) {
				numActsLocal[i] = 4; 
			}
			
			numActsLocal[numLayersLocal - 1] = 1;
			
			for (int i = 0; i < numSuperficialNNModels; i++) {
				localNNs.add(new NeuralNetworkRegression(clusterComputingMode
							, acceleratingComputingMode
							, numLayersLocal
							, numActsLocal
							, new NonlinearCGOptimizer()));
			}

			// Configure the abstract NN model layer.
			int numLayersA = abstractLayerDepth;
			int[] numActsA = new int[numLayersA];
			
			numActsA[0] = numSuperficialNNModels;
			
			for (int i = 1; i < numLayersA - 1; i++) {
				numActsA[i] = Math.max(5, numSuperficialNNModels / 10); 
			}
			
			numActsA[numLayersA - 1] = 1;
			
			abstractNN = new NeuralNetworkRegression(clusterComputingMode
					, acceleratingComputingMode
					, numLayersA
					, numActsA
					, new NonlinearCGOptimizer());
		}
		
		// Train the model.
		public void train(MMRFInfo info, int trainingMode) {
			
			// Input data is assumed to be valid.
			
			if (trainingMode == TRAINING_MODE_STATIC) {
				
				// Load trained NN parameters from a base64 encoded string.
				loadNNParamsFromBase64String(nnParamsBase64Str);
				return;
			}
			
			// Train the superficial NN model layer.
			int numSamples = info.expr_avg.colLength();
			List<Matrix> geneExpPartialXs = new ArrayList<Matrix>();
			List<Matrix> geneVarPartialXs = new ArrayList<Matrix>();
			List<Matrix> geneMutPartialXs = new ArrayList<Matrix>();
			
			Matrix X = null;
			Matrix Y = info.survival_risk;
			
			for (int i = 0; i < numSuperficialNNModels; i++) {
				
				// Get partial training data.
				int[] rangeX = {1 + i * partialNumGenes
				                , (i + 1) * partialNumGenes
				                , 1
				                , numSamples};
				
				// Gene expression.
				X = info.expr_avg.getSubMatrix(rangeX);
				AbstractNeuralNetwork.TrainingResult re 
					= partialExpRiskNNs.get(i).train(X, Y, Lambda, 0, 0, 0, numIter, false, false, 1.0);
				geneExpPartialXs.add(X);
				
				// Gene variation.
				X = info.expr_diff.getSubMatrix(rangeX);
				AbstractNeuralNetwork.TrainingResult rv 
					= partialVarRiskNNs.get(i).train(X, Y, Lambda, 0, 0, 0, numIter, false, false, 1.0);
				geneVarPartialXs.add(X);
				
				// Gene mutation.
				X = info.mutation.getSubMatrix(rangeX);
				AbstractNeuralNetwork.TrainingResult rm 
					= partialMutRiskNNs.get(i).train(X, Y, Lambda, 0, 0, 0, numIter, false, false, 1.0);
				geneMutPartialXs.add(X);
			}
						
			// Train the local NN model layer.
			List<Matrix> localXs = new ArrayList<Matrix>();
			
			// Calculate training data for the local NN model layer.
			for (int i = 0; i < numSuperficialNNModels; i++) {
				
				// Gene expression.
				Matrix X2  
					= partialExpRiskNNs.get(i).predict(geneExpPartialXs.get(i));
				
				// Gene variation.
				X2 = X2.verticalAdd(partialVarRiskNNs.get(i).predict(geneVarPartialXs.get(i)));
				
				// Gene mutation.
				X2 = X2.verticalAdd(partialMutRiskNNs.get(i).predict(geneMutPartialXs.get(i)));
				
				localXs.add(X2);
				
				// Train.
				AbstractNeuralNetwork.TrainingResult rl 
					= localNNs.get(i).train(X2, Y, Lambda, 0, 0, 0, numIter, false, false, 1.0);
			}
		
			// Train the abstract NN model layer.
			// Calculate training data for the abstract NN model layer.
			Matrix X3 = null;
			X3 = localNNs.get(0).predict(localXs.get(0));
			
			for (int i = 1; i < numSuperficialNNModels; i++) {
				X3 = X3.verticalAdd(localNNs.get(i).predict(localXs.get(i)));
			}
	
			// Train.
			AbstractNeuralNetwork.TrainingResult ra 
				= abstractNN.train(X3, Y, Lambda, 0, 0, 0, numIter, false, false, 1.0);
			
			// Save trained NN parameters into a base64 encoded string.
			if (DEBUG) {
				saveNNParamsToBase64String();
			}
		}
		
		// Load trained NN parameters from a base64 encoded string.
		public void loadNNParamsFromBase64String(String v) {
			
			// An input parameter is assumed to be valid.
			
			// Load.
			String[] vs = v.split(",");
			int count = 0;
			
			// Superficial NN models layer.
			for (int i = 0; i < numSuperficialNNModels; i++) {
				Utility.decodeNNParamsToBase64(vs[count++], partialExpRiskNNs.get(i));
				Utility.decodeNNParamsToBase64(vs[count++], partialVarRiskNNs.get(i));
				Utility.decodeNNParamsToBase64(vs[count++], partialMutRiskNNs.get(i));
			}
			
			// Local NN models layer.
			for (int i = 0; i < numSuperficialNNModels; i++) {
				Utility.decodeNNParamsToBase64(vs[count++], localNNs.get(i));
			}
			
			// Abstract NN model layer.
			Utility.decodeNNParamsToBase64(vs[count], abstractNN);
		}
		
		// Save trained NN parameters into a base64 encoded string.
		public String saveNNParamsToBase64String() {
			
			// Create a NN params. base64 encoded string builder.
			StringBuilder t = new StringBuilder();
			
			// Superficial NN models layer.
			for (int i = 0; i < numSuperficialNNModels; i++) {
				t.append(Utility.encodeNNParamsToBase64(partialExpRiskNNs.get(i)));
				t.append(",");
				t.append(Utility.encodeNNParamsToBase64(partialVarRiskNNs.get(i)));
				t.append(",");
				t.append(Utility.encodeNNParamsToBase64(partialMutRiskNNs.get(i)));
				t.append(",");
			}
			
			// Local NN models layer.
			for (int i = 0; i < numSuperficialNNModels; i++) {
				t.append(Utility.encodeNNParamsToBase64(localNNs.get(i)));
				t.append(",");
			}
			
			// Abstract NN model layer.
			t.append(Utility.encodeNNParamsToBase64(abstractNN));
			
			// Save.
			/*
			try {
				FileWriter fw = new FileWriter("nnParamsBase64.dat");
				fw.write(t.toString());
				fw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			*/
			
			return t.toString();
		}
		
		// Predict.
		public Matrix predict(MMRFInfo info) {
			
			// Input data is assumed to be valid.
			
			// Get partial data.
			int numSamples = info.expr_avg.colLength();
			List<Matrix> geneExpPartialXs = new ArrayList<Matrix>();
			List<Matrix> geneVarPartialXs = new ArrayList<Matrix>();
			List<Matrix> geneMutPartialXs = new ArrayList<Matrix>();
			
			Matrix X = null;
			Matrix Y = info.survival_risk;
			
			for (int i = 0; i < numSuperficialNNModels; i++) {
				
				// Get partial training data.
				int[] rangeX = {1 + i * partialNumGenes
				                , (i + 1) * partialNumGenes
				                , 1
				                , numSamples};
				
				// Gene expression.
				X = info.expr_avg.getSubMatrix(rangeX);
				geneExpPartialXs.add(X);
				
				// Gene variation.
				X = info.expr_diff.getSubMatrix(rangeX);
				geneVarPartialXs.add(X);
				
				// Gene mutation.
				X = info.mutation.getSubMatrix(rangeX);
				geneMutPartialXs.add(X);
			}
						
			// Conduct the feedforward of the local NN model layer.
			List<Matrix> localXs = new ArrayList<Matrix>();
						
			for (int i = 0; i < numSuperficialNNModels; i++) {
				
				// Gene expression.
				Matrix X2  
					= partialExpRiskNNs.get(i).predict(geneExpPartialXs.get(i));
				
				// Gene variation.
				X2 = X2.verticalAdd(partialVarRiskNNs.get(i).predict(geneVarPartialXs.get(i)));
				
				// Gene mutation.
				X2 = X2.verticalAdd(partialMutRiskNNs.get(i).predict(geneMutPartialXs.get(i)));
				
				localXs.add(X2);
			}
		
			// Conduct the feedforward of the abstract NN model layer.
			Matrix X3 = null;
			X3 = localNNs.get(0).predict(localXs.get(0));
			
			for (int i = 1; i < numSuperficialNNModels; i++) {
				X3 = X3.verticalAdd(localNNs.get(i).predict(localXs.get(i)));
			}
	
			return abstractNN.predict(X3);
		}
	}
	
	// Gene expression average values.
	private double[] geneExpAvgs = null;
	
	// Gene variation average values.
	private double[] geneVarAvgs = null;
	
	// MM progress predictor.
	public MMProgressPredictor mmpp = null;
	
	// Pivot genes.
	/* 
['FABP5',  
 'PDHA1',
 'TRIP13',
 'AIM2',
 'LARS2',
 'OPN3',
 'ASPM',
 'CCT2',
 'UBE2I',
 'LAS1L',
 'BIRC5',
 'RFC4',
 'PFN1',
 'ILF3',
 'IFI16',
 'TBRG4',
 'ENO1',
 'DSG2',
 'EXOSC4',
 'TAGLN2',
 'RUVBL1',
 'ALDOA',
 'CPSF3',
 'LGALS1',
 'RAD18',
 'SNX5',
 'PSMD4',
 'RAN',
 'KIF14',
 'CBX3',
 'TMPO',
 'WEE1',
 'ROBO1',
 'TCOF1',
 'YWHAZ',
 'GNG10',
 'PNPLA4',
 'AHCYL1',
 'EVI5',
 'CTBS',
 'UBE2R2',
 'FUCA1',
 'LTBP1',
 'TRIM33']
	                           	 */
	public int[] pivotGenes = {5009,
	                           11907,
	                           17132,
	                           452,
	                           8213,
	                           11162,
	                           1096,
	                           2534,
	                           17457,
	                           8215,
	                           1470,
	                           13509,
	                           12006,
	                           7313,
	                           7126,
	                           16176,
	                           4751,
	                           4387,
	                           4966,
	                           16063,
	                           14028,
	                           532,
	                           3455,
	                           8305,
	                           13231,
	                           15381,
	                           12943,
	                           13279,
	                           7835,
	                           2310,
	                           16796,
	                           17962,
	                           13777,
	                           16233,
	                           18127,
	                           6152,
	                           12406,
	                           432,
	                           4938,
	                           3615,
	                           17470,
	                           5725,
	                           8713,
	                           17079};

	
	/*
	['KIF14',
	 'SLC19A1',
	 'YWHAZ',
	 'TMPO',
	 'TBRG4',
	 'AIM2',
	 'ASPM',
	 'AHCYL1',
	 'CTBS',
	 'LTBP1']


	public int[] pivotGenes = {7835, 14686, 18127, 16796, 16176, 452, 1096, 432, 3615, 8713};
		*/
	
	public final static int numSymNeighborGenes = 5;
	public int NUM_GENES = pivotGenes.length * (1 + 2 * numSymNeighborGenes);
	
	public double survivalRiskWeight = 100000.0;
	
	/**
	 * Train data.
	 * @param test_type
	 * @param expr_avg
	 * @param expr_diff
	 * @param mutation
	 * @param prog_obs_time
	 * @return
	 */
	public int trainingData(int test_type
			, String[] expr_avg
			, String[] expr_diff
			, String[] mutation
			, String[] prog_obs_time) {
		
		// Parse and preprocess data.
		MMRFInfo pivotInfo = parseProcDataForTraining(expr_avg, expr_diff, mutation, prog_obs_time);
		
		// Create a MM progress predictor.
		mmpp = new MMProgressPredictor(3, 3);
		
		// Train the MM progress predictor.
		mmpp.train(pivotInfo, TRAINING_MODE_DYNAMIC);
		
		return 0;
	}
		
	/**
	 * Test data.
	 * @param expr_avg
	 * @param expr_diff
	 * @param mutation
	 * @return
	 */
	public int[] testingData(String[] expr_avg
			, String[] expr_diff
			, String[] mutation) {
		
		// Parse and preprocess data.
		MMRFInfo pivotInfo = parseProcDataForTesting(expr_avg, expr_diff, mutation);
		
		// Test.
		Matrix resultM = mmpp.predict(pivotInfo);
		
		// Get a survival risk rank-ordered list in decreasing order.
		return getSurvivalRiskRanks(resultM);
	}
	
	// Get a survival risk rank-ordered list in decreasing order.
	public int[] getSurvivalRiskRanks(Matrix resultM) {
		
		// An input parameter is assumed to be valid.
		
		// Sort a survival risk list.
		double[] survivalRisks = resultM.unrolledVector();
		int[] ids = new int[survivalRisks.length];
		
		for (int i = 0; i < survivalRisks.length; i++) {
			ids[i] = i + 1;
		}
		
		for (int i = 0; i < survivalRisks.length - 1; i++) {
			for (int j = i; j < survivalRisks.length; j++) {
				if (survivalRisks[i] < survivalRisks[j]) {
					double temp = survivalRisks[i];
					survivalRisks[i] = survivalRisks[j];
					survivalRisks[j] = temp;
					int tempId = ids[i];
					ids[i] = ids[j];
					ids[j] = tempId;
				}
			}
		}
		
		// Assign ranks.
		int[] ranks = new int[survivalRisks.length];
		
		for (int i = 0; i < survivalRisks.length; i++) {
			ranks[ids[i] - 1] = i + 1;
		}
		
		return ranks;
	}
	
	// Parse and preprocess data for training.
	public MMRFInfo parseProcDataForTraining(String[] expr_avg
			, String[] expr_diff
			, String[] mutation
			, String[] prog_obs_time) {
		
		// Parse.
		MMRFInfo info = new MMRFInfo();
		int numSamples = expr_avg.length;
		boolean isFirstP = true;
		boolean isFirstE = true;
		boolean isFirstV = true;
		boolean isFirstM = true;
		
		// Initialize average values.
		geneExpAvgs = new double[TOTAL_NUM_GENES];
		geneVarAvgs = new double[TOTAL_NUM_GENES];
		
		int validNumSamples = 0;
		
		// Skip header.
		for (int i = 1; i < numSamples; i++) {
			
			// prog_obs_time and survival_risk.
			{
				String[] prog_obs_times = prog_obs_time[i].split(",");
				
				// Check whether there is either progression time or observation time.
				if (prog_obs_times.length == 0)
					continue;
				
				double[] vals = new double[2];
				int count = 0;
				
				for (String vStr : prog_obs_times) {
					vals[count++] = vStr.isEmpty() == true ? Double.NaN : Double.valueOf(vStr);
				}
								
				if (isFirstP) {
					info.prog_obs_time = new Matrix(2, 1, vals);
					
					// Calculate survival risk.
					if (!Double.isNaN(vals[0])) {
						info.survival_risk = new Matrix(1, 1, 1.0 / vals[0] * survivalRiskWeight);
					} else {
						info.survival_risk = new Matrix(1, 1, 1.0 / vals[1] * survivalRiskWeight);
					}
					
					isFirstP = false;
				} else {
					info.prog_obs_time = info.prog_obs_time.horizontalAdd(new Matrix(2, 1, vals));
					
					// Calculate survival risk.
					if (!Double.isNaN(vals[0])) {
						info.survival_risk = info.survival_risk.horizontalAdd(new Matrix(1, 1, 1.0 / vals[0] * survivalRiskWeight));
					} else {
						info.survival_risk = info.survival_risk.horizontalAdd(new Matrix(1, 1, 1.0 / vals[1] * survivalRiskWeight));
					}
				}
			}

			// expr_avg.
			{
				String[] expr_avgs = expr_avg[i].split(",");
				double[] vals = new double[TOTAL_NUM_GENES];
				int count = 0;
				
				for (String vStr : expr_avgs) {
					vals[count] = vStr.isEmpty() == true ? Double.NaN : Double.valueOf(vStr);
					geneExpAvgs[count] += Double.isNaN(vals[count]) ? 0 : vals[count];
					count++;
				}
				
				if (isFirstE) {
					info.expr_avg = new Matrix(TOTAL_NUM_GENES, 1, vals);
					isFirstE = false;
				} else {
					info.expr_avg = info.expr_avg.horizontalAdd(new Matrix(TOTAL_NUM_GENES, 1, vals));
				}
			}
			
			// expr_diff.
			{
				String[] expr_diffs = expr_diff[i].split(",");
				double[] vals = new double[TOTAL_NUM_GENES];
				int count = 0;
				
				for (String vStr : expr_diffs) {
					vals[count] = vStr.isEmpty() == true ? Double.NaN : Double.valueOf(vStr);
					geneVarAvgs[count] += Double.isNaN(vals[count]) ? 0 : vals[count];
					count++;
				}
				
				if (isFirstV) {
					info.expr_diff = new Matrix(TOTAL_NUM_GENES, 1, vals);
					isFirstV = false;
				} else {
					info.expr_diff = info.expr_diff.horizontalAdd(new Matrix(TOTAL_NUM_GENES, 1, vals));
				}
			}
			
			// mutation.
			{
				String[] mutations = mutation[i].split(",");
				double[] vals = new double[TOTAL_NUM_GENES];
				int count = 0;
				
				for (String vStr : mutations) {
					vals[count] = vStr.isEmpty() == true ? 0.0 : Double.valueOf(vStr);
					count++;
				}
				
				if (isFirstM) {
					info.mutation = new Matrix(TOTAL_NUM_GENES, 1, vals);
					isFirstM = false;
				} else {
					info.mutation = info.mutation.horizontalAdd(new Matrix(TOTAL_NUM_GENES, 1, vals));
				}
			}
			
			validNumSamples++;
		}
	
		// Preprocess.
		// Get averaged values for each factor.
		for (int i = 0; i < TOTAL_NUM_GENES; i++) {
			geneExpAvgs[i] /= (double)validNumSamples;
			geneVarAvgs[i] /= (double)validNumSamples;
		}
		
		// Filter outliers.
		// TODO
		
		// Fill missing factors.
		// It is assumed that a row with a NaN value has total NaN values.
		for (int i = 0; i < validNumSamples; i++) {
			if (Double.isNaN(info.expr_avg.getVal(1, i + 1))) {
				for (int j = 0; j < TOTAL_NUM_GENES; j++) {
					info.expr_avg.setVal(j + 1, i + 1, geneExpAvgs[j]);
				}
			}
			
			if (Double.isNaN(info.expr_diff.getVal(1, i + 1))) {
				for (int j = 0; j < TOTAL_NUM_GENES; j++) {
					info.expr_diff.setVal(j + 1, i + 1, geneVarAvgs[j]);
				}
			}
		}
		
		// Extract pivot genes and relevant neighbor genes.
		MMRFInfo pivotInfo = new MMRFInfo();
		
		int[] range = {pivotGenes[0] + 1 - numSymNeighborGenes,
				pivotGenes[0] + 1 + numSymNeighborGenes, 1, validNumSamples};
		pivotInfo.expr_avg = info.expr_avg.getSubMatrix(range);
		pivotInfo.expr_diff = info.expr_diff.getSubMatrix(range);
		pivotInfo.mutation = info.mutation.getSubMatrix(range);
		
		pivotInfo.prog_obs_time = info.prog_obs_time;
		pivotInfo.survival_risk = info.survival_risk;
		
		for (int i = 1; i < pivotGenes.length; i++) {
			int[] iRange = {pivotGenes[i] + 1 - numSymNeighborGenes,
					pivotGenes[i] + 1 + numSymNeighborGenes, 1, validNumSamples};
			pivotInfo.expr_avg = pivotInfo.expr_avg.verticalAdd(info.expr_avg.getSubMatrix(iRange));
			pivotInfo.expr_diff = pivotInfo.expr_diff.verticalAdd(info.expr_diff.getSubMatrix(iRange));
			pivotInfo.mutation = pivotInfo.mutation.verticalAdd(info.mutation.getSubMatrix(iRange));
		}
		
		return pivotInfo;
	}

	// Parse and preprocess data for testing.
	public MMRFInfo parseProcDataForTesting(String[] expr_avg
			, String[] expr_diff
			, String[] mutation) {
		
		// Parse.
		MMRFInfo info = new MMRFInfo();
		int numSamples = expr_avg.length;
		boolean isFirstP = true;
		boolean isFirstE = true;
		boolean isFirstV = true;
		boolean isFirstM = true;
		
		// Skip header.
		for (int i = 1; i < numSamples; i++) {
			
			// expr_avg.
			{
				String[] expr_avgs = expr_avg[i].split(",");
				double[] vals = new double[TOTAL_NUM_GENES];
				int count = 0;
				
				for (String vStr : expr_avgs) {
					vals[count] = vStr.isEmpty() == true ? Double.NaN : Double.valueOf(vStr);
					count++;
				}
				
				if (isFirstE) {
					info.expr_avg = new Matrix(TOTAL_NUM_GENES, 1, vals);
					isFirstE = false;
				} else {
					info.expr_avg = info.expr_avg.horizontalAdd(new Matrix(TOTAL_NUM_GENES, 1, vals));
				}
			}
			
			// expr_diff.
			{
				String[] expr_diffs = expr_diff[i].split(",");
				double[] vals = new double[TOTAL_NUM_GENES];
				int count = 0;
				
				for (String vStr : expr_diffs) {
					vals[count] = vStr.isEmpty() == true ? Double.NaN : Double.valueOf(vStr);
					count++;
				}
				
				if (isFirstV) {
					info.expr_diff = new Matrix(TOTAL_NUM_GENES, 1, vals);
					isFirstV = false;
				} else {
					info.expr_diff = info.expr_diff.horizontalAdd(new Matrix(TOTAL_NUM_GENES, 1, vals));
				}
			}
			
			// mutation.
			{
				String[] mutations = mutation[i].split(",");
				double[] vals = new double[TOTAL_NUM_GENES];
				int count = 0;
				
				for (String vStr : mutations) {
					vals[count] = vStr.isEmpty() == true ? 0.0 : Double.valueOf(vStr);
					count++;
				}
				
				if (isFirstM) {
					info.mutation = new Matrix(TOTAL_NUM_GENES, 1, vals);
					isFirstM = false;
				} else {
					info.mutation = info.mutation.horizontalAdd(new Matrix(TOTAL_NUM_GENES, 1, vals));
				}
			}
		}
	
		// Preprocess.		
		// Filter outliers.
		// TODO
		
		// Fill missing factors.
		// It is assumed that a row with a NaN value has total NaN values.
		numSamples = numSamples - 1;
		
		for (int i = 0; i < numSamples; i++) {
			if (Double.isNaN(info.expr_avg.getVal(1, 1 + i))) {
				for (int j = 0; j < TOTAL_NUM_GENES; j++) {
					info.expr_avg.setVal(j + 1, i + 1, geneExpAvgs[j]);
				}
			}
			
			if (Double.isNaN(info.expr_diff.getVal(1, 1 + i))) {
				for (int j = 0; j < TOTAL_NUM_GENES; j++) {
					info.expr_diff.setVal(j + 1, i + 1, geneVarAvgs[j]);
				}
			}
		}
		
		// Extract pivot genes and relevant neighbor genes.
		MMRFInfo pivotInfo = new MMRFInfo();
		
		int[] range = {pivotGenes[0] + 1 - numSymNeighborGenes,
				pivotGenes[0] + 1 + numSymNeighborGenes, 1, numSamples};
		pivotInfo.expr_avg = info.expr_avg.getSubMatrix(range);
		pivotInfo.expr_diff = info.expr_diff.getSubMatrix(range);
		pivotInfo.mutation = info.mutation.getSubMatrix(range);
				
		for (int i = 1; i < pivotGenes.length; i++) {
			int[] iRange = {pivotGenes[i] + 1 - numSymNeighborGenes,
					pivotGenes[i] + 1 + numSymNeighborGenes, 1, numSamples};
			pivotInfo.expr_avg = pivotInfo.expr_avg.verticalAdd(info.expr_avg.getSubMatrix(iRange));
			pivotInfo.expr_diff = pivotInfo.expr_diff.verticalAdd(info.expr_diff.getSubMatrix(iRange));
			pivotInfo.mutation = pivotInfo.mutation.verticalAdd(info.mutation.getSubMatrix(iRange));
		}
		
		return pivotInfo;
	}	
}
