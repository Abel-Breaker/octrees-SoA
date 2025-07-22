//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNELSQUARE_HPP
#define KERNELSQUARE_HPP

#include "Kernel2D.hpp"
#include "util.hpp"

class KernelSquare : public Kernel2D
{
	public:
	KernelSquare(const Point& center, const double radius) : Kernel2D(center, radius) {}
	KernelSquare(const Point& center, const Vector& radii) : Kernel2D(center, radii) {}

	[[nodiscard]] bool isInside(const Point& p) const override
	/**
 * @brief Checks if a given point lies inside the kernel
 * @param p
 * @return
 */
	{
		return onInterval(p.getX(), boxMin().getX(), boxMax().getX()) &&
		       onInterval(p.getY(), boxMin().getY(), boxMax().getY());
	}

	/**
	 * @brief For the boxIntersect functions, we check if the box passed is inside,
	 * overlaps or is outside the bounding box of the square kernel
	 * 
	 * @returns IntersectionJudgement, a enum value signaling each of the three conditions
	*/
	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const double radius) const override
	{
		// Box bounds
		const double highX = center.getX() + radius, lowX = center.getX() - radius;
		const double highY = center.getY() + radius, lowY = center.getY() - radius;

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX(); 
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY();

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY ||
			lowX > boxMaxX 	|| lowY > boxMaxY) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY &&
		   lowX >= boxMinX 	&& lowY >= boxMinY) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}

	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const Vector& radii) const override
	{
		// Box bounds
		const double highX = center.getX() + radii.getX(), lowX = center.getX() - radii.getX();
		const double highY = center.getY() + radii.getY(), lowY = center.getY() - radii.getY();

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX();
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY(); 

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY ||
			lowX > boxMaxX 	|| lowY > boxMaxY) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY &&
		   lowX >= boxMinX 	&& lowY >= boxMinY) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}
};

#endif /* end of include guard: KERNELSQUARE_HPP */
