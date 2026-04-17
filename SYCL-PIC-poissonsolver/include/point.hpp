/**
 * @brief Legacy point structure for file reading
 */
struct PointOld {
    double x, y;
};

/**
 * @brief Enhanced point structure with position and velocity
 */
struct Point{
    double x, y, z, vx, vy, vz;
    int pointer;
};

struct Field{
    double fx;
    double fy;
};

struct Geometry{
    int x1,y1,x2,y2;
    double potential;
};
