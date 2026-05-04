model BouncingBallsN
  // N-ball hybrid bouncing-ball model with N-dependent initial conditions.
  //
  // This version uses separate when clauses and keeps exactly one reinit(...)
  // statement in each when clause.
  //
  // To avoid conflicting reinitializations, the event conditions that can
  // reinitialize the same state variable are made mutually exclusive:
  //
  //   x[i]  : leftHit[i] and rightHit[i] are mutually exclusive.
  //   y[i]  : floorHit[i] and ceilingHit[i] are mutually exclusive.
  //   vx[i] : leftHit[i], rightHit[i], and ballVxHit[i] are mutually exclusive.
  //   vy[i] : floorHit[i], ceilingHit[i], and ballVyHit[i] are mutually exclusive.
  //
  // Wall events use strict outside-the-domain guards, e.g.,
  //   x[i] < x_min
  // instead of
  //   x[i] <= x_min.
  // After the wall event clamps x[i] to x_min, the condition becomes false,
  // which helps avoid repeated wall-event firing.

  parameter Integer N(min=1) = 3 "Number of balls";

  parameter Real g = 9.81 "Gravity";
  parameter Real e_g = 0.5 "Wall restitution coefficient";
  parameter Real e_b = 0.6 "Ball-ball restitution coefficient";
  parameter Real d_sq = 0.05 "Squared center-distance threshold for ball-ball collision";

  parameter Real y_max = 10.0;
  parameter Real y_min = 0.0;
  parameter Real x_max = 10.0;
  parameter Real x_min = 0.0;

  parameter Real vx_max0 = 0.6
    "Maximum absolute initial horizontal velocity";
  parameter Real vy_max0 = 0.55
    "Maximum absolute initial vertical velocity";

  function numberOfColumns
    input Integer N;
    output Integer nCols;
  algorithm
    nCols := 1;
    while nCols*nCols < N loop
      nCols := nCols + 1;
    end while;
  end numberOfColumns;

  function numberOfRows
    input Integer N;
    input Integer nCols;
    output Integer nRows;
  algorithm
    nRows := div(N + nCols - 1, nCols);
  end numberOfRows;

  function initialXUpperHalf
    input Integer N;
    input Real x_min;
    input Real x_max;
    output Real x0[N];
  protected
    Integer nCols;
    Integer col;
  algorithm
    nCols := numberOfColumns(N);

    for i in 1:N loop
      col := mod(i - 1, nCols) + 1;
      x0[i] := x_min + (col - 0.5)*(x_max - x_min)/nCols;
    end for;
  end initialXUpperHalf;

  function initialYUpperHalf
    input Integer N;
    input Real y_min;
    input Real y_max;
    output Real y0[N];
  protected
    Integer nCols;
    Integer nRows;
    Integer row;
    Real y_mid;
  algorithm
    nCols := numberOfColumns(N);
    nRows := numberOfRows(N, nCols);
    y_mid := y_min + 0.5*(y_max - y_min);

    for i in 1:N loop
      row := div(i - 1, nCols) + 1;
      y0[i] := y_mid + (row - 0.5)*(y_max - y_mid)/nRows;
    end for;
  end initialYUpperHalf;

  function initialVx
    input Integer N;
    input Real vx_max0;
    output Real vx0[N];
  protected
    Integer nCols;
    Integer col;
  algorithm
    nCols := numberOfColumns(N);

    for i in 1:N loop
      col := mod(i - 1, nCols) + 1;

      vx0[i] :=
        if nCols > 1 then
          -vx_max0 + 2*vx_max0*(col - 1)/(nCols - 1)
        else
          0.0;
    end for;
  end initialVx;

  function initialVy
    input Integer N;
    input Real vy_max0;
    output Real vy0[N];
  protected
    Integer nCols;
    Integer nRows;
    Integer row;
  algorithm
    nCols := numberOfColumns(N);
    nRows := numberOfRows(N, nCols);

    for i in 1:N loop
      row := div(i - 1, nCols) + 1;

      vy0[i] :=
        if nRows > 1 then
          -vy_max0 + 2*vy_max0*(row - 1)/(nRows - 1)
        else
          -vy_max0;
    end for;
  end initialVy;

  parameter Real x0[N]  = initialXUpperHalf(N, x_min, x_max);
  parameter Real y0[N]  = initialYUpperHalf(N, y_min, y_max);
  parameter Real vx0[N] = initialVx(N, vx_max0);
  parameter Real vy0[N] = initialVy(N, vy_max0);

  Real x[N](start=x0, each fixed=true);
  Real y[N](start=y0, each fixed=true);
  Real vx[N](start=vx0, each fixed=true);
  Real vy[N](start=vy0, each fixed=true);

protected
  Boolean leftHit[N];
  Boolean rightHit[N];
  Boolean floorHit[N];
  Boolean ceilingHit[N];

  Boolean pairHit[N, N];
  Boolean ballHit[N];

  // Collision-event filters used to avoid conflicting reinit calls.
  Boolean ballVxHit[N];
  Boolean ballVyHit[N];

equation
  // Continuous dynamics.
  for i in 1:N loop
    der(x[i])  = vx[i];
    der(y[i])  = vy[i];
    der(vx[i]) = 0;
    der(vy[i]) = -g;
  end for;

  // Strict outside-the-domain wall guards.
  for i in 1:N loop
    leftHit[i]    = x[i] < x_min;
    rightHit[i]   = x[i] > x_max;
    floorHit[i]   = y[i] < y_min;
    ceilingHit[i] = y[i] > y_max;
  end for;

  // Pairwise ball-collision guards.
  for i in 1:N loop
    for j in 1:N loop
      pairHit[i, j] =
        if i <> j then
          (x[i] - x[j])^2 + (y[i] - y[j])^2 - d_sq < 0
        else
          false;
    end for;
  end for;

  for i in 1:N loop
    ballHit[i] = sum(if pairHit[i, j] then 1 else 0 for j in 1:N) > 0;

    // For vx[i], ball collision is allowed only if no x-wall event is active.
    // Thus leftHit[i], rightHit[i], and ballVxHit[i] are mutually exclusive.
    ballVxHit[i] = ballHit[i] and not leftHit[i] and not rightHit[i];

    // For vy[i], ball collision is allowed only if no y-wall event is active.
    // Thus floorHit[i], ceilingHit[i], and ballVyHit[i] are mutually exclusive.
    ballVyHit[i] = ballHit[i] and not floorHit[i] and not ceilingHit[i];
  end for;

  // -------------------------------------------------------------------------
  // Separate mutually exclusive when clauses.
  // Each when clause contains exactly one reinit(...) statement.
  // -------------------------------------------------------------------------

  for i in 1:N loop

    // Position clamps: x.
    when leftHit[i] then
      reinit(x[i], x_min);
    end when;

    when rightHit[i] then
      reinit(x[i], x_max);
    end when;

    // Position clamps: y.
    when floorHit[i] then
      reinit(y[i], y_min);
    end when;

    when ceilingHit[i] then
      reinit(y[i], y_max);
    end when;

    // Wall velocity resets: vx.
    when leftHit[i] then
      reinit(vx[i], -e_g*pre(vx[i]));
    end when;

    when rightHit[i] then
      reinit(vx[i], -e_g*pre(vx[i]));
    end when;

    // Wall velocity resets: vy.
    when floorHit[i] then
      reinit(vy[i], -e_g*pre(vy[i]));
    end when;

    when ceilingHit[i] then
      reinit(vy[i], -e_g*pre(vy[i]));
    end when;

    // Ball-ball velocity resets, filtered to avoid conflicting with wall
    // reinitializations of the same velocity component.
    when ballVxHit[i] then
      reinit(vx[i], -e_b*pre(vx[i]));
    end when;

    when ballVyHit[i] then
      reinit(vy[i], -e_b*pre(vy[i]));
    end when;

  end for;

  annotation (
    experiment(StartTime=0, StopTime=10, Tolerance=1e-6, Interval=0.001),
    Documentation(info="<html>
      <p>N-ball hybrid bouncing-ball model.</p>
      <p>All initial conditions x0, y0, vx0, and vy0 are generated as functions of N.</p>
      <p>This version uses separate mutually exclusive when clauses with exactly one reinit statement per when clause.</p>
    </html>")
  );
end BouncingBallsN;
