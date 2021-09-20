type Aabb = crate::Aabb<i32>;
type Point = crate::Vector3<i32>;

#[test]
fn test_aabb_intersection() {
    let aabb1 = Aabb::from_extents(Point::new(0, 0, 0), Point::new(3, 3, 3));
    let aabb2 = Aabb::from_extents(Point::new(-1, -1, -1), Point::new(2, 2, 2));

    assert_eq!(
        Aabb::from_extents(Point::new(0, 0, 0), Point::new(2, 2, 2)),
        aabb1.get_intersection(aabb2)
    );
}
