*DASUM
      FUNCTION DASUM(N,DX,INCX) RESULT(DASUMR)
C***Begin Prologue  DASUM
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A3A
C***Keywords  Add,BLAS,REAL(wp),Linear Algebra,Magnitude,Sum,
C             Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. OF TEXAS)
C           Krogh, F. T., (JPL)
C***Purpose  Sum of Magnitudes of D.P. Vector Components
C***Description
C                B L A S  Subprogram
C    Description of parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C     --Output--
C    DASUM  REAL(wp) result (Zero IF N .LE. 0)
C     Returns sum of magnitudes of REAL(wp) DX.
C     DASUM = Sum from 0 to N-1 of DABS(DX(1+I*INCX))
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms For FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines called  (none)
C***End Prologue  DASUM

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      INTEGER
     &   INCX,N

C...Array arguments
      REAL(wp)
     &   DX(*)

C...Result
      REAL(wp)
     &   DASUMR

C...Local scalars
      INTEGER
     &   I,M,MP1,NS

C...Intrinsic functions
      INTRINSIC
     &   DABS,MOD


C***First executable statement  DASUM


      DASUMR = 0.E0_wp
      IF(N.LE.0)RETURN
      IF(INCX.EQ.1)GOTO 20

C        Code for increments not equal to 1.

      NS = N*INCX
          DO 10 I=1,NS,INCX
          DASUMR = DASUMR + DABS(DX(I))
   10     CONTINUE
      RETURN

C        Code for increments equal to 1.

C        Clean-up loop so remaining vector length is a multiple of 6.

   20 M = MOD(N,6)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
         DASUMR = DASUMR + DABS(DX(I))
   30 CONTINUE
      IF( N .LT. 6 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,6
         DASUMR = DASUMR + DABS(DX(I)) + DABS(DX(I+1)) + DABS(DX(I+2))
     1   + DABS(DX(I+3)) + DABS(DX(I+4)) + DABS(DX(I+5))
   50 CONTINUE
      RETURN
      END
*DAXPY
      SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
C***Begin Prologue  DAXPY
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A7
C***Keywords  BLAS,REAL(wp),Linear Algebra,Triad,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  D.P Computation Y = A*X + Y
C***Description
C                B L A S  Subprogram
C    Description of parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DA  REAL(wp) scalar multiplier
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C       DY  REAL(wp) vector with N elements
C     INCY  Storage spacing between elements of DY
C     --Output--
C       DY  REAL(wp) result (unchanged IF N .LE. 0)
C     Overwrite REAL(wp) DY with REAL(wp) DA*DX + DY.
C     For I = 0 to N-1, replace  DY(LY+I*INCY) with DA*DX(LX+I*INCX) +
C       DY(LY+I*INCY), where LX = 1 IF INCX .GE. 0, ELSE LX = (-INCX)*N
C       and LY is defined in a similar way using INCY.
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines called  (none)
C***End Prologue  DAXPY

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      REAL(wp)
     &   DA
      INTEGER
     &   INCX,INCY,N

C...Array arguments
      REAL(wp)
     &   DX(*),DY(*)

C...Local scalars
      INTEGER
     &   I,IX,IY,M,MP1,NS

C...Intrinsic functions
      INTRINSIC
     &   MOD


C***First executable statement  DAXPY


      IF(N.LE.0.OR.DA.EQ.0.E0_wp) RETURN
      IF(INCX.EQ.INCY) THEN
        IF(INCX-1.LT.0) THEN
          GOTO 5
        ELSE IF (INCX-1.EQ.0) THEN
          GOTO 20
        ELSE IF (INCX-1.GT.0) THEN
          GOTO 60
        END IF
      END IF
    5 CONTINUE

C        Code for nonequal or nonpositive increments.

      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DY(IY) = DY(IY) + DA*DX(IX)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN

C        Code for both increments equal to 1


C        Clean-up loop so remaining vector length is a multiple of 4.

   20 M = MOD(N,4)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DY(I) = DY(I) + DA*DX(I)
   30 CONTINUE
      IF( N .LT. 4 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,4
        DY(I) = DY(I) + DA*DX(I)
        DY(I + 1) = DY(I + 1) + DA*DX(I + 1)
        DY(I + 2) = DY(I + 2) + DA*DX(I + 2)
        DY(I + 3) = DY(I + 3) + DA*DX(I + 3)
   50 CONTINUE
      RETURN

C        Code for equal, positive, nonunit increments.

   60 CONTINUE
      NS = N*INCX
          DO 70 I=1,NS,INCX
          DY(I) = DA*DX(I) + DY(I)
   70     CONTINUE
      RETURN
      END
*DCOPY
      SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
C***Begin Prologue  DCOPY
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A5
C***Keywords  BLAS,Copy,REAL(wp),Linear Algebra,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  D.P. Vector Copy Y = X
C***Description
C                B L A S  Subprogram
C    Description of parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C       DY  REAL(wp) vector with N elements
C     INCY  Storage spacing between elements of DY
C     --Output--
C       DY  Copy of vector DX (unchanged if N .LE. 0)
C     Copy REAL(wp) DX to REAL(wp) DY.
C     For I = 0 to N-1, copy DX(LX+I*INCX) to DY(LY+I*INCY),
C     where LX = 1 if INCX .GE. 0, else LX = (-INCX)*N, and LY is
C     defined in a similar way using INCY.
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines called  (none)
C***End Prologue  DCOPY

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      INTEGER
     &   INCX,INCY,N

C...Array arguments
      REAL(wp)
     &   DX(*),DY(*)

C...Local scalars
      INTEGER
     &   I,IX,IY,M,MP1,NS

C...Intrinsic functions
      INTRINSIC
     &   MOD


C***First executable statement  DCOPY


      IF(N.LE.0)RETURN
      IF(INCX.EQ.INCY) THEN
        IF(INCX-1.LT.0) THEN
          GOTO 5
        ELSE IF(INCX-1.EQ.0) THEN
          GOTO 20
        ELSE IF(INCX-1.GT.0) THEN
          GOTO 60
        END IF
      END IF
    5 CONTINUE

C        Code for unequal or nonpositive increments.

      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DY(IY) = DX(IX)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN

C        Code for both increments equal to 1


C        Clean-up loop so remaining vector length is a multiple of 7.

   20 M = MOD(N,7)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DY(I) = DX(I)
   30 CONTINUE
      IF( N .LT. 7 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,7
        DY(I) = DX(I)
        DY(I + 1) = DX(I + 1)
        DY(I + 2) = DX(I + 2)
        DY(I + 3) = DX(I + 3)
        DY(I + 4) = DX(I + 4)
        DY(I + 5) = DX(I + 5)
        DY(I + 6) = DX(I + 6)
   50 CONTINUE
      RETURN

C        Code for equal, positive, nonunit increments.

   60 CONTINUE
      NS=N*INCX
          DO 70 I=1,NS,INCX
          DY(I) = DX(I)
   70     CONTINUE
      RETURN
      END
*DDOT
      FUNCTION DDOT(N,DX,INCX,DY,INCY) RESULT(DDOTR)
C***Begin Prologue  DDOT
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A4
C***Keywords  BLAS,REAL(wp),Inner Product,Linear Algebra,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  D.P. Inner Product of D.P. Vectors
C***Description
C                B L A S  Subprogram
C    Description of parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C       DY  REAL(wp) vector with N elements
C     INCY  Storage spacing between elements of DY
C     --Output--
C     DDOT  REAL(wp) dot product (zero if N .LE. 0)
C     returns the dot product of REAL(wp) DX and DY.
C     DDOT = SUM for I = 0 to N-1 of  DX(LX+I*INCX) * DY(LY+I*INCY)
C     where LX = 1 if INCX .GE. 0, else LX = (-INCX)*N, and LY is
C     defined in a similar way using INCY.
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines called  (none)
C***End Prologue  DDOT

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      INTEGER
     &   INCX,INCY,N

C...Array arguments
      REAL(wp)
     &   DX(*),DY(*)

C...Result
      REAL(wp)
     &   DDOTR

C...Local scalars
      INTEGER
     &   I,IX,IY,M,MP1,NS

C...Intrinsic functions
      INTRINSIC
     &   MOD


C***First executable statement  DDOT


      DDOTR = 0.E0_wp
      IF(N.LE.0)RETURN
      IF(INCX.EQ.INCY) THEN
        IF(INCX-1.LT.0) THEN
          GOTO 5
        ELSE IF(INCX-1.EQ.0) THEN
          GOTO 20
        ELSE IF(INCX-1.GT.0) THEN
          GOTO 60
        END IF
      END IF
    5 CONTINUE

C         Code for unequal or nonpositive increments.

      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
         DDOTR = DDOTR + DX(IX)*DY(IY)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN

C        Code for both increments equal to 1.


C        Clean-up loop so remaining vector length is a multiple of 5.

   20 M = MOD(N,5)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
         DDOTR = DDOTR + DX(I)*DY(I)
   30 CONTINUE
      IF( N .LT. 5 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
         DDOTR = DDOTR + DX(I)*DY(I) + DX(I+1)*DY(I+1) +
     1   DX(I + 2)*DY(I + 2) + DX(I + 3)*DY(I + 3) + DX(I + 4)*DY(I + 4)
   50 CONTINUE
      RETURN

C         Code for positive equal increments .NE.1.

   60 CONTINUE
      NS = N*INCX
          DO 70 I=1,NS,INCX
          DDOTR = DDOTR + DX(I)*DY(I)
   70     CONTINUE
      RETURN
      END
*DNRM2
      FUNCTION DNRM2(N,DX,INCX) RESULT(DNRM2R)
C***Begin Prologue  DNRM2
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A3B
C***Keywords  BLAS,REAL(wp),Euclidean,L2,Length,Linear Algebra,
C             Norm,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           KROGH, F. T., (JPL)
C***Purpose  Euclidean Length (L2 Norm) of D.P. Vector
C***Description
C                B L A S  Subprogram
C    Description of parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C     --Output--
C    DNRM2  REAL(wp) result (zero if N .LE. 0)
C     Euclidean norm of the N-vector stored in DX() with storage
C     increment INCX .
C     If    N .LE. 0 return with result = 0.
C     If N .GE. 1 then INCX must be .GE. 1
C           C.L. Lawson, 1978 Jan 08
C     Four Phase Method     Using two built-in constants that are
C     hopefully applicable to all machines.
C         CUTLO = Maximum of  DSQRT(U/EPS)  over all known machines.
C         CUTHI = Minimum of  DSQRT(V)      over all known machines.
C     where
C         EPS = smallest no. such that EPS + 1. .GT. 1.
C         U   = smallest positive no.   (underflow limit)
C         V   = largest  no.            (overflow  limit)
C     Brief outline of algorithm..
C     Phase 1    Scans zero components.
C     Move to Phase 2 when a component is nonzero and .LE. CUTLO
C     Move to Phase 3 when a component is .GT. CUTLO
C     Move to Phase 4 when a component is .GE. CUTHI/M
C     where M = N for X() REAL and M = 2*N for COMPLEX.

C     Values for CUTLO and CUTHI..
C     From the environmental parameters listed in the IMSL converter
C     document the limiting values are as follows..
C     CUTLO, S.P.   U/EPS = 2**(-102) for  Honeywell.  Close seconds are
C                   UNIVAC and DEC at 2**(-103)
C                   thus CUTLO = 2**(-51) = 4.44089E-16_wp
C     CUTHI, S.P.   V = 2**127 for UNIVAC, Honeywell, and DEC.
C                   thus CUTHI = 2**(63.5) = 1.30438E19_wp
C     CUTLO, D.P.   U/EPS = 2**(-67) for Honeywell and DEC.
C                   thus CUTLO = 2**(-33.5) = 8.23181E-11_wp
C     CUTHI, D.P.   Same as S.P.  CUTHI = 1.30438E19_wp
C     DATA CUTLO, CUTHI / 8.232E-11_wp,  1.304E19_wp /
C     DATA CUTLO, CUTHI / 4.441E-16_wp,  1.304E19_wp /
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines called  (none)
C***End Prologue  DNRM2

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      INTEGER
     &   INCX,N

C...Array arguments
      REAL(wp)
     &   DX(*)

C...Result
      REAL(wp)
     &   DNRM2R

C...Local scalars
      REAL(wp)
     &   CUTHI,CUTLO,HITEST,ONE,SUM,XMAX,ZERO
      INTEGER
     &   I,J,NEXT,NN

C...Intrinsic functions
      INTRINSIC
     &   DABS,DSQRT,FLOAT

C...Data statements
      DATA
     &   ZERO,ONE/0.0E0_wp,1.0E0_wp/
      DATA
     &   CUTLO,CUTHI/8.232E-11_wp,1.304E19_wp/


C***First executable statement  DNRM2


      XMAX = ZERO
      IF(N .GT. 0) GO TO 10
         DNRM2R  = ZERO
         GO TO 300

   10 NEXT=30
      SUM = ZERO
      NN = N * INCX
C                                                 Begin main loop
      I = 1
   20 IF (NEXT.EQ.30) THEN; GOTO 30; END IF
      IF (NEXT.EQ.50) THEN; GOTO 50; END IF
      IF (NEXT.EQ.70) THEN; GOTO 70; END IF
      IF (NEXT.EQ.110) THEN; GOTO 110; END IF
   30 IF( DABS(DX(I)) .GT. CUTLO) GO TO 85
      NEXT=50
      XMAX = ZERO

C                        Phase 1.  Sum is zero

   50 IF( DX(I) .EQ. ZERO) GO TO 200
      IF( DABS(DX(I)) .GT. CUTLO) GO TO 85

C                                Prepare for Phase 2.
      NEXT=70
      GO TO 105

C                                Prepare for Phase 4.

  100 I = J
      NEXT=110
      SUM = (SUM / DX(I)) / DX(I)
  105 XMAX = DABS(DX(I))
      GO TO 115

C                   Phase 2.  Sum is small.
C                             Scale to avoid destructive underflow.

   70 IF( DABS(DX(I)) .GT. CUTLO ) GO TO 75

C                     Common code for Phases 2 and 4.
C                     In Phase 4 sum is large.  Scale to avoid overflow.

  110 IF( DABS(DX(I)) .LE. XMAX ) GO TO 115
         SUM = ONE + SUM * (XMAX / DX(I))**2
         XMAX = DABS(DX(I))
         GO TO 200

  115 SUM = SUM + (DX(I)/XMAX)**2
      GO TO 200


C                  Prepare for Phase 3.

   75 SUM = (SUM * XMAX) * XMAX


C     For REAL OR D.P. set HITEST = CUTHI/N
C     For COMPLEX      set HITEST = CUTHI/(2*N)

   85 HITEST = CUTHI/FLOAT( N )

C                   Phase 3.  Sum is mid-range.  No scaling.

      DO 95 J =I,NN,INCX
        IF(DABS(DX(J)) .GE. HITEST) GO TO 100
        SUM = SUM + DX(J)**2
   95 CONTINUE
      DNRM2R = DSQRT( SUM )
      GO TO 300

  200 CONTINUE
      I = I + INCX
      IF ( I .LE. NN ) GO TO 20

C              End of main loop.

C              Compute square root and adjust for scaling.

      DNRM2R = XMAX * DSQRT(SUM)
  300 CONTINUE
      RETURN
      END
*DROT
      SUBROUTINE DROT(N,DX,INCX,DY,INCY,DC,DS)
C***Begin Prologue  DROT
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A8
C***Keywords  BLAS,Givens Rotation,Linear Algebra,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  Apply D.P. Givens Rotation
C***Description
C                B L A S  Subprogram
C    Description of Parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C       DY  REAL(wp) vector with N elements
C     INCY  Storage spacing between elements of DY
C       DC  D.P. element of rotation matrix
C       DS  D.P. element of rotation matrix
C     --Output--
C       DX  Rotated vector (unchanged if N .LE. 0)
C       DY  Rotated vector (unchanged if N .LE. 0)
C     Multiply the 2 x 2 matrix  ( DC DS) times the 2 x N matrix (DX**T)
C                                (-DS DC)                        (DY**T)
C     where **T indicates transpose.  The elements of DX are in
C     DX(LX+I*INCX), I = 0 to N-1, where LX = 1 if INCX .GE. 0, else
C     LX = (-INCX)*N, and similarly for DY using LY and INCY.
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines Called  (NONE)
C***End Prologue  DROT

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      REAL(wp)
     &   DC,DS
      INTEGER
     &   INCX,INCY,N

C...Array arguments
      REAL(wp)
     &   DX(*),DY(*)

C...Local scalars
      REAL(wp)
     &   ONE,W,Z,ZERO
      INTEGER
     &   I,KX,KY,NSTEPS

C...Data statements
      DATA
     &   ZERO,ONE/0.E0_wp,1.E0_wp/


C***First executable statement  DROT


      IF(N .LE. 0 .OR. (DS .EQ. ZERO .AND. DC .EQ. ONE)) GO TO 40
      IF(.NOT. (INCX .EQ. INCY .AND. INCX .GT. 0)) GO TO 20

           NSTEPS=INCX*N
           DO 10 I=1,NSTEPS,INCX
                W=DX(I)
                Z=DY(I)
                DX(I)=DC*W+DS*Z
                DY(I)=-DS*W+DC*Z
   10           CONTINUE
           GO TO 40

   20 CONTINUE
           KX=1
           KY=1

           IF(INCX .LT. 0) KX=1-(N-1)*INCX
           IF(INCY .LT. 0) KY=1-(N-1)*INCY

           DO 30 I=1,N
                W=DX(KX)
                Z=DY(KY)
                DX(KX)=DC*W+DS*Z
                DY(KY)=-DS*W+DC*Z
                KX=KX+INCX
                KY=KY+INCY
   30           CONTINUE
   40 CONTINUE

      RETURN
      END
*DROTG
      SUBROUTINE DROTG(DA,DB,DC,DS)
C***Begin Prologue  DROTG
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1B10
C***Keywords  BLAS,Givens Rotation,Linear Algebra,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  Construct D.P. Plane Givens Rotation
C***Description
C                B L A S  Subprogram
C    Description of Parameters
C     --Input--
C       DA  REAL(wp) scalar
C       DB  REAL(wp) scalar
C     --Output--
C       DA  REAL(wp) result R
C       DB  REAL(wp) result Z
C       DC  REAL(wp) result
C       DS  REAL(wp) result
C     Designed By C. L. Lawson, JPL, 1977 Sept 08
C     Construct the Givens Transformation
C         ( DC  DS )
C     G = (        ) ,    DC**2 + DS**2 = 1 ,
C         (-DS  DC )
C     which zeros the second entry of the 2-vector  (DA,DB)**T .
C     the quantity R = (+/-)DSQRT(DA**2 + DB**2) overwrites DA in
C     storage.  The value of DB is overwritten by a value Z which
C     allows DC and DS to be recovered by the following algorithm.
C           If Z=1  set  DC=0.E0_wp  and  DS=1.E0_wp
C           If DABS(Z) .LT. 1  set  DC=DSQRT(1-Z**2)  and  DS=Z
C           If DABS(Z) .GT. 1  set  DC=1/Z  and  DS=DSQRT(1-DC**2)
C     Normally, the subprogram DROT(N,DX,INCX,DY,INCY,DC,DS) will
C     next be called to apply the transformation to a 2 by N matrix.
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines Called  (None)
C***End Prologue  DROTG

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      REAL(wp)
     &   DA,DB,DC,DS

C...Local scalars
      REAL(wp)
     &   R,U,V

C...Intrinsic functions
      INTRINSIC
     &   DABS,DSQRT


C***First executable statement  DROTG


      IF (DABS(DA) .LE. DABS(DB)) GO TO 10

C     *** Here DABS(DA) .GT. DABS(DB) ***

      U = DA + DA
      V = DB / U

C     Note that U and R have the sign of DA

      R = DSQRT(.25E0_wp + V**2) * U

C     Note that DC is positive

      DC = DA / R
      DS = V * (DC + DC)
      DB = DS
      DA = R
      RETURN

C *** Here DABS(DA) .LE. DABS(DB) ***

   10 IF (DB .EQ. 0.E0_wp) GO TO 20
      U = DB + DB
      V = DA / U

C     Note that U and R have the sign of DB
C     (R is immediately stored in DA)

      DA = DSQRT(.25E0_wp + V**2) * U

C     Note that DS is positive

      DS = DB / DA
      DC = V * (DS + DS)
      IF (DC .EQ. 0.E0_wp) GO TO 15
      DB = 1.E0_wp / DC
      RETURN
   15 DB = 1.E0_wp
      RETURN

C *** Here DA = DB = 0.E0_wp ***

   20 DC = 1.E0_wp
      DS = 0.E0_wp
      RETURN

      END
*DSCAL
      SUBROUTINE DSCAL(N,DA,DX,INCX)
C***Begin Prologue  DSCAL
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A6
C***Keywords  BLAS,Linear Algebra,Scale,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  D.P. Vector Scale X = A*X
C***Description
C                B L A S  Subprogram
C    Description of Parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DA  REAL(wp) scale factor
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C     --Output--
C       DX  REAL(wp) result (unchanged if N.LE.0)
C     Replace REAL(wp) DX by REAL(wp) DA*DX.
C     For I = 0 to N-1, replace DX(1+I*INCX) with  DA * DX(1+I*INCX)
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines Called  (None)
C***End Prologue  DSCAL

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      REAL(wp)
     &   DA
      INTEGER
     &   INCX,N

C...Array arguments
      REAL(wp)
     &   DX(*)

C...Local scalars
      INTEGER
     &   I,M,MP1,NS

C...Intrinsic functions
      INTRINSIC
     &   MOD


C***First executable statement  DSCAL


      IF(N.LE.0)RETURN
      IF(INCX.EQ.1)GOTO 20

C        Code for increments not equal to 1.

      NS = N*INCX
          DO 10 I = 1,NS,INCX
          DX(I) = DA*DX(I)
   10     CONTINUE
      RETURN

C        Code for increments equal to 1.


C        Clean-up loop so remaining vector length is a multiple of 5.

   20 M = MOD(N,5)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DX(I) = DA*DX(I)
   30 CONTINUE
      IF( N .LT. 5 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
        DX(I) = DA*DX(I)
        DX(I + 1) = DA*DX(I + 1)
        DX(I + 2) = DA*DX(I + 2)
        DX(I + 3) = DA*DX(I + 3)
        DX(I + 4) = DA*DX(I + 4)
   50 CONTINUE
      RETURN
      END
*DSWAP
      SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
C***Begin Prologue  DSWAP
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A5
C***Keywords  BLAS,REAL(wp),Interchange,Linear Algebra,Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  Interchange D.P. vectors
C***Description
C                B L A S  Subprogram
C    Description of Parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C       DY  REAL(wp) vector with N elements
C     INCY  Storage spacing between elements of DY
C     --Output--
C       DX  Input vector DY (unchanged if N .LE. 0)
C       DY  Input vector DX (unchanged if N .LE. 0)
C     Interchange REAL(wp) DX and REAL(wp) DY.
C     For I = 0 TO N-1, interchange  DX(LX+I*INCX) and DY(LY+I*INCY),
C     where LX = 1 if INCX .GE. 0, else LX = (-INCX)*N, and LY is
C     defined in a similar way using INCY.
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines Called  (None)
C***End Prologue  DSWAP

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      INTEGER
     &   INCX,INCY,N

C...Array arguments
      REAL(wp)
     &   DX(*),DY(*)

C...Local scalars
      REAL(wp)
     &   DTEMP1,DTEMP2,DTEMP3
      INTEGER
     &   I,IX,IY,M,MP1,NS

C...Intrinsic functions
      INTRINSIC
     &   MOD


C***First executable statement  DSWAP


      IF(N.LE.0)RETURN
      IF(INCX.EQ.INCY) THEN
        IF(INCX-1.LT.0) THEN
          GOTO 5
        ELSE IF(INCX-1.EQ.0) THEN
          GOTO 20
        ELSE IF(INCX-1.GT.0) THEN
          GOTO 60
        END IF
      END IF
    5 CONTINUE

C       Code for unequal or nonpositive increments.

      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DTEMP1 = DX(IX)
        DX(IX) = DY(IY)
        DY(IY) = DTEMP1
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN

C       Code for both increments equal to 1


C       Clean-up loop so remaining vector length is a multiple of 3.

   20 M = MOD(N,3)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DTEMP1 = DX(I)
        DX(I) = DY(I)
        DY(I) = DTEMP1
   30 CONTINUE
      IF( N .LT. 3 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,3
        DTEMP1 = DX(I)
        DTEMP2 = DX(I+1)
        DTEMP3 = DX(I+2)
        DX(I) = DY(I)
        DX(I+1) = DY(I+1)
        DX(I+2) = DY(I+2)
        DY(I) = DTEMP1
        DY(I+1) = DTEMP2
        DY(I+2) = DTEMP3
   50 CONTINUE
      RETURN
   60 CONTINUE

C     Code for equal, positive, nonunit increments.

      NS = N*INCX
        DO 70 I=1,NS,INCX
        DTEMP1 = DX(I)
        DX(I) = DY(I)
        DY(I) = DTEMP1
   70   CONTINUE
      RETURN
      END
*IDAMAX
      FUNCTION IDAMAX(N,DX,INCX) RESULT(IDAMAXR)
C***Begin Prologue  IDAMAX
C***Date Written   791001   (YYMMDD)
C***Revision Date  820801   (YYMMDD)
C***Category No.  D1A2
C***Keywords  BLAS,REAL(wp),Linear Algebra,Maximum Component,
C             Vector
C***Author  Lawson, C. L., (JPL)
C           Hanson, R. J., (SNLA)
C           Kincaid, D. R., (U. of Texas)
C           Krogh, F. T., (JPL)
C***Purpose  Find largest component of D.P. vector
C***Description
C                B L A S  Subprogram
C    Description of parameters
C     --Input--
C        N  Number of elements in input vector(s)
C       DX  REAL(wp) vector with N elements
C     INCX  Storage spacing between elements of DX
C     --Output--
C   IDAMAX  Smallest index (zero if N .LE. 0)
C     Find smallest index of maximum magnitude of REAL(wp) DX.
C     IDAMAX =  first I, I = 1 to N, to minimize  ABS(DX(1-INCX+I*INCX)
C***References  Lawson C.L., Hanson R.J., Kincaid D.R., Krogh F.T.,
C                 *Basic Linear Algebra Subprograms for FORTRAN Usage*,
C                 Algorithm No. 539, Transactions on Mathematical
C                 Software, Volume 5, Number 3, September 1979, 308-323
C***Routines Called  (None)
C***End Prologue  IDAMAX

C...Used modules
      use odrpack_kinds, only: wp

C...Scalar arguments
      INTEGER
     &   INCX,N

C...Array arguments
      REAL(wp)
     &   DX(*)

C...Result
      INTEGER
     &   IDAMAXR

C...Local scalars
      REAL(wp)
     &   DMAX,XMAG
      INTEGER
     &   I,II,NS

C...Intrinsic functions
      INTRINSIC
     &   DABS


C***First executable statement  IDAMAX


      IDAMAXR = 0
      IF(N.LE.0) RETURN
      IDAMAXR = 1
      IF(N.LE.1)RETURN
      IF(INCX.EQ.1)GOTO 20

C        Code for increments not equal to 1.

      DMAX = DABS(DX(1))
      NS = N*INCX
      II = 1
          DO 10 I = 1,NS,INCX
          XMAG = DABS(DX(I))
          IF(XMAG.LE.DMAX) GO TO 5
          IDAMAXR = II
          DMAX = XMAG
    5     II = II + 1
   10     CONTINUE
      RETURN

C        Code for increments equal to 1.

   20 DMAX = DABS(DX(1))
      DO 30 I = 2,N
          XMAG = DABS(DX(I))
          IF(XMAG.LE.DMAX) GO TO 30
          IDAMAXR = I
          DMAX = XMAG
   30 CONTINUE
      RETURN
      END
