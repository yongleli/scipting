!__doc__input file CONTCAR
!__doc__outfile x.res
!this is a function making poscar contcar to x.res
!
!
PROGRAM MAIN
!!!!!!!!!!!!!!!!!!!!!define!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
REAL                    ::A,B,C,HASH_R,alp,bat,gam,pi
REAL                    ::A1(3),A2(3),A3(3),POS(3),C1,C2
INTEGER,ALLOCATABLE     ::ATOM_R(:)
CHARACTER*4             ::HASH_C,ATOM_NAME
CHARACTER*1             ::order_n
CHARACTER*2,ALLOCATABLE ::ATOM_C(:)
INTEGER                 ::N_TYPE,I,J,C3,M,N
!!!!!!!!!!!!!!!!!!!!Conculate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
OPEN(1,FILE='CONTCAR')
OPEN(2,FILE='x.res')
READ(1,*)HASH_C
READ(1,*)HASH_R
READ(1,*)A1(:)
READ(1,*)A2(:)
READ(1,*)A3(:)
A=SQRT(A1(1)**2+A1(2)**2+A1(3)**2)
B=SQRT(A2(1)**2+A2(2)**2+A2(3)**2)
C=SQRT(A3(1)**2+A3(2)**2+A3(3)**2)
alp=acos(dot_product(A2,A3)/B*C)
bet=acos(dot_product(A1,A3)/A*C)
gam=acos(dot_product(A1,A2)/A*B)
pi=2*acos(0.0)
alp=alp*180/pi
bet=bet*180/pi
gam=gam*180/pi
C1=1.0
C2=0.0
C3=-1.0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!rite(*,*)A,B,C,ALP,BET,GAM,pi!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!read!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WRITE(*,*)'PLZ INPUT ATOM TYPE:'
READ(*,*)N_TYPE
ALLOCATE(ATOM_R(N_TYPE))
ALLOCATE(ATOM_C(N_TYPE))
READ(1,*)ATOM_C(:)
READ(1,*)ATOM_R(:)
READ(1,*)HASH_C
N=0
!!!!!!!!!!!!!!!Write!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
write(2,'(A4,X,2(A2))')'TITL',ATOM_C(:)
WRITE(2,'(A4,2X,7(F7.4,2X))')'CELL',C2,A,B,C,alp,bet,gam
WRITE(2,'(A4,X,I3)')'LATT',C3
WRITE(2,'(A4,X,2(A2,X))')'SFAC',ATOM_C(:)
DO I=1,N_TYPE
   DO J=1,ATOM_R(I)
      READ(1,*)POS(:)
      M=J+N
      WRITE(order_n,'(I1)')M
      ATOM_NAME=TRIM(ATOM_C(I))//order_n
      WRITE(2,'(A4,X,I2,2X,5(F7.5,2X))')ATOM_NAME,i,POS(:),C1,C2
    ENDDO
      N=N+ATOM_R(I)
ENDDO

WRITE(2,'(A3)')'END'
END
