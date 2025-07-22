//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNELCUBE_HPP
#define KERNELCUBE_HPP

#include "Kernel3D.hpp"
#include "util.hpp"

class KernelCube : public Kernel3D
{
	public:
	KernelCube(const Point& center, const double radius) : Kernel3D(center, radius) {}
	KernelCube(const Point& center, const Vector& radii) : Kernel3D(center, radii) {}

	[[nodiscard]] bool isInside(const Point& p) const override
	/**
 * @brief Checks if a given point lies inside the kernel
 * @param p
 * @return
 */
	{
		return onInterval(p.getX(), boxMin().getX(), boxMax().getX()) &&
		       onInterval(p.getY(), boxMin().getY(), boxMax().getY()) &&
		       onInterval(p.getZ(), boxMin().getZ(), boxMax().getZ());
	};

	/**
	 * @brief For the boxIntersect functions, we check if the box passed is inside,
	 * overlaps or is outside the bounding box of the square cube
	 * 
	 * @returns IntersectionJudgement, a enum value signaling each of the three conditions
	*/
	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const double radius) const override
	{
		// Box bounds
		const double highX = center.getX() + radius, lowX = center.getX() - radius;
		const double highY = center.getY() + radius, lowY = center.getY() - radius;
		const double highZ = center.getZ() + radius, lowZ = center.getZ() - radius;

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX(); 
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY(); 
		const double boxMaxZ = boxMax().getZ(), boxMinZ = boxMin().getZ();

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY 	|| highZ < boxMinZ || 
			lowX > boxMaxX 	|| lowY > boxMaxY 	|| lowZ > boxMaxZ) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY && highZ <= boxMaxZ &&
		   lowX >= boxMinX 	&& lowY >= boxMinY 	&& lowZ >= boxMinZ) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}

	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const Vector& radii) const override
	{
		// Box bounds
		const double highX = center.getX() + radii.getX(), lowX = center.getX() - radii.getX();
		const double highY = center.getY() + radii.getY(), lowY = center.getY() - radii.getY();
		const double highZ = center.getZ() + radii.getZ(), lowZ = center.getZ() - radii.getZ();

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX(); 
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY(); 
		const double boxMaxZ = boxMax().getZ(), boxMinZ = boxMin().getZ();

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY 	|| highZ < boxMinZ || 
			lowX > boxMaxX 	|| lowY > boxMaxY 	|| lowZ > boxMaxZ) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY && highZ <= boxMaxZ &&
		   lowX >= boxMinX 	&& lowY >= boxMinY 	&& lowZ >= boxMinZ) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}


};


#endif /* end of include guard: KERNELCUBE_HPP */
