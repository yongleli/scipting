program main
USE MSIMSL
real a,ax,ay,bx,by,x1,y1,x2,y2,Rij,dx,dy,theta,dxr,dyr
real x(50),y(50),xr(50),yr(50),kx(153),ky(153),kgx(3),kgy(3)
real k(3,3),u(3,3),v(3,3),kt(3,3),rd,id
complex Dxy(4,3,3),D(6,6),EIG(6) !AA--D1,
integer N,nk,t,tt
a=2.462
if(1)then
!-------a-b=60 du-----------
ax=a*cosd(30.0);ay=-a*sind(30.0);bx=a*cosd(30.0);by=a*sind(30.0)
x1=1.0/3;y1=1.0/3  !A atom
x2=2.0/3;y2=2.0/3  !b atom
print*,"     -b"
print*,"   -    -"
print*," -        -"    
print*,"- ==A--B== -"
print*," -        -"
print*,"   -    -"
print*,"     -a"
else
!-------a along x, a,b=120 du-------
ax=a;ay=0;bx=-a*cosd(60.0);by=a*sind(60.0)
x1=2.0/3;y1=1.0/3  !A atom
x2=1.0/3;y2=2.0/3  !b atom
print*,"  _________"
print*,"  \b       \"
print*,"   \  B     \"
print*,"    \     A  \"
print*,"     \________\a"
!--------------
endif
!通过平移产生对称包围原子a,b的原子阵列
N=0
do i=-2,2
  do j=-2,2
N=N+1;x(N)=x1+i;y(N)=y1+j;x(N+25)=x2+i;y(N+25)=y2+j
  enddo
enddo
open(11,FILE="real.dat")
do i=1,50
!print*,i,X(i),Y(i),'psg'
xr(i)=ax*x(i)+bx*y(i);yr(i)=ay*x(i)+by*y(i)
write(11,"(f6.3,2x,f6.3)")xr(i),yr(i)
enddo
!通过平移产生对称包围原子a,b的原子阵列
!在高对称线上产生k点
kgx(1)=0;kgy(1)=0;kgx(2)=0.5;kgy(2)=0;kgx(3)=1.0/3;kgy(3)=2.0/3
do i=1,51
if(0)then  !wrong
kx(i)=(i-1)*(kgx(2)-kgx(1))/50;ky(i)=(i-1)*(kgy(2)-kgy(1))/50  !1--->2
kx(i+51)=(i-1)*(kgx(3)-kgx(2))/50;ky(i)=(i-1)*(kgy(3)-kgy(2))/50  !2--->3
kx(i+102)=(i-1)*(kgx(1)-kgx(3))/50;ky(i)=(i-1)*(kgy(1)-kgy(3))/50  !3--->1
else  !correct
kx(i)=(i-1)*(kgx(2)-kgx(1))/50;ky(i)=(i-1)*(kgy(2)-kgy(1))/50  !1--->2
kx(i+51)=kgx(2)+(i-1)*(kgx(3)-kgx(2))/50;ky(i+51)=kgy(2)+(i-1)*(kgy(3)-kgy(2))/50  !2--->3
kx(i+102)=kgx(3)+(i-1)*(kgx(1)-kgx(3))/50;ky(i+102)=kgy(3)+(i-1)*(kgy(1)-kgy(3))/50  !3--->1
endif
enddo
!-----------solve each K points----------------------
open(111,file="dd.dat")
open(113,file="BZZ.dat")
do nk=1,153
!kx(1)=0.5;ky(1)=0
D=0
    Dxy=0
do i=13,38,25
    do j=1,50
	!-------------求Rij,theta------------------------
	dx=xr(j)-xr(i);dy=yr(j)-yr(i)  !absolutely displacement
	Rij=sqrt(dx**2+dy**2)

	if(dx==0)then
	    if(dy>0)theta=acos(-1.0)/2
        if(dy<0)theta=-acos(-1.0)/2
		else
	theta=atan(abs(dy/dx))
	if(dx>0.and.dy>0)theta=theta
	if(dx>0.and.dy<0)theta=-theta
	if(dx<0.and.dy>0)theta=acos(-1.0)-theta
	if(dx<0.and.dy<0)theta=-acos(-1.0)+theta
	endif
!!----KK-------
jl=0
if(abs(Rij-1.426)<0.01)jl=1
if(abs(Rij-a)<0.01)jl=2
if(abs(Rij-2.84)<0.01)jl=3
if(abs(Rij-3.758)<0.01)jl=4
k=0
!if(i==13.and.j==31)print*,i,j,jl
!if(i==13.and.j==31)print*,Rij,jl
!if(i==13.and.j==31)pause
select case(jl)
case(1)
k(1,1)=36.50;k(2,2)=24.50;k(3,3)=9.82
case(2)
k(1,1)=8.80;k(2,2)=-3.23;k(3,3)=-0.40
case(3)
k(1,1)=3.00;k(2,2)=-5.25;k(3,3)=0.15
case(4)
k(1,1)=-1.92;k(2,2)=2.29;k(3,3)=-0.58
case default
k=0
cycle
end select
u=0;v=0
u(1,1)=cos(theta);u(1,2)=sin(theta);u(2,1)=-sin(theta);u(2,2)=cos(theta);u(3,3)=1.0
v(1,1)=cos(theta);v(1,2)=-sin(theta);v(2,1)=sin(theta);v(2,2)=cos(theta);v(3,3)=1.0
!!!!!!!!!!!!!!!!!!!!V*K--->Kt
do  ii=1,3
  do jj=1,3
  kt(ii,jj)=v(ii,1)*k(1,jj)+v(ii,2)*k(2,jj)+v(ii,3)*k(3,jj)
  enddo
enddo
!!!!!!!!!!!!!!!!!!!!Kt*U--->K:描述I,J原子的相互作用矩阵
do  ii=1,3
  do jj=1,3
  k(ii,jj)=kt(ii,1)*u(1,jj)+kt(ii,2)*u(2,jj)+kt(ii,3)*u(3,jj)
  enddo
enddo
!!!!!!!

!--------------solve smart DAA,DAB,DBA,DBB----------------
dx=x(i)-x(j);dy=y(i)-y(j)  !relative displacement
if(0)then
print*,i,x(i),y(i)
print*,j,x(j),y(j)
print*,i,j,theta*180/acos(-1.0),"--PSG"
print*,dx,dy
write(*,"(3F12.8)") k
pause
endif
do ii=1,3
  do jj=1,3
if(i==13)then  !!AA----------------
     if(j>25)tt=0
     if(j<=25)tt=1
     rd=K(ii,jj)-tt*K(ii,jj)*cos(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     id=-tt*K(ii,jj)*sin(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     Cab=cmplx(rd,id)
	 Dxy(1,II,jj)=Dxy(1,II,jj)+cmplx(rd,id) 
  endif !!AA----------------

if(i==13)then  !!AB----------------
     if(j<=25)tt=0
     if(j>25)tt=1
     rd=-tt*K(ii,jj)*cos(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     id=-tt*K(ii,jj)*sin(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     Cab=cmplx(rd,id)
	 Dxy(2,II,jj)=Dxy(2,II,jj)+cmplx(rd,id)
  endif !!AB---------------- 

if(i==38)then  !!BA----------------
     if(j<=25)tt=1
     if(j>25)tt=0
     rd=-tt*K(ii,jj)*cos(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     id=-tt*K(ii,jj)*sin(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     Cab=cmplx(rd,id)
	 Dxy(3,II,jj)=Dxy(3,II,jj)+cmplx(rd,id)
  endif !!AB----------------

  if(i==38)then  !!BB----------------
     if(j<=25)tt=0
     if(j>25)tt=1
     rd=K(ii,jj)-tt*K(ii,jj)*cos(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     id=-tt*K(ii,jj)*sin(dx*kx(nk)*2*acos(-1.0)+dy*ky(nk)*2*acos(-1.0))
     Cab=cmplx(rd,id)
	 Dxy(4,II,jj)=Dxy(4,II,jj)+cmplx(rd,id) 
  endif !!BB----------------
  enddo
  enddo

!--------------solve smart DAA,DAB,DBA,DBB----------------
do ii=1,3
do jj=1,3
D(ii,jj)=Dxy(1,ii,jj)
D(ii,jj+3)=Dxy(2,ii,jj)
D(ii+3,jj)=Dxy(3,ii,jj)
D(ii+3,jj+3)=Dxy(4,ii,jj)
enddo
enddo  !-------construct the Dij with DAA,DAB,DBA,DBB---------------
	enddo  !!!! j=1,50
enddo     !!-i=13,38,25
!-------------------------------------------------------
write(111,*)nk,"Kpoints Dij"
write(111,'(6F8.3)')real(D)
call EVLCG(6,D,6,EIG)
Write(113,"(i5,2x,6f12.6)")nk,real(EIG)  !!out put BZZ
enddo !做每个K点的动力学矩阵
!----------------------
end