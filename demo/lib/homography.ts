export type CalibrationPoint = { x: number; y: number };

export type CalibrationValidation = {
  valid: boolean;
  areaRatio: number;
  message: string;
};

const COURT_POINTS: CalibrationPoint[] = [
  { x: 0, y: 0 },
  { x: 6.1, y: 0 },
  { x: 6.1, y: 13.4 },
  { x: 0, y: 13.4 },
];

function solveLinearSystem(rows: number[][]): number[] {
  const size = rows.length;
  const matrix = rows.map((row) => [...row]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    }
    if (Math.abs(matrix[pivot][column]) < 1e-10) throw new Error("角点几何退化，无法计算 Homography");
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    const divisor = matrix[column][column];
    for (let index = column; index <= size; index += 1) matrix[column][index] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = matrix[row][column];
      for (let index = column; index <= size; index += 1) matrix[row][index] -= factor * matrix[column][index];
    }
  }
  return matrix.map((row) => row[size]);
}

export function computeHomography(source: CalibrationPoint[]): number[][] {
  if (source.length !== 4) throw new Error("需要四个球场角点");
  const rows: number[][] = [];
  source.forEach(({ x, y }, index) => {
    const { x: u, y: v } = COURT_POINTS[index];
    rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    rows.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  });
  const values = solveLinearSystem(rows);
  return [values.slice(0, 3), values.slice(3, 6), [...values.slice(6, 8), 1]];
}

export function projectPoint(matrix: number[][], point: CalibrationPoint): CalibrationPoint {
  const denominator = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2];
  return {
    x: (matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2]) / denominator,
    y: (matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2]) / denominator,
  };
}

export function validateCalibration(points: CalibrationPoint[], width: number, height: number): CalibrationValidation {
  if (points.length !== 4) return { valid: false, areaRatio: 0, message: `还需点击 ${4 - points.length} 个角点` };
  if (width <= 0 || height <= 0) return { valid: false, areaRatio: 0, message: "视频分辨率不可用" };
  if (points.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x < 0 || point.y < 0 || point.x > width || point.y > height)) {
    return { valid: false, areaRatio: 0, message: "角点必须位于视频画面内" };
  }

  const signedArea = points.reduce((sum, point, index) => {
    const next = points[(index + 1) % points.length];
    return sum + point.x * next.y - next.x * point.y;
  }, 0) / 2;
  const areaRatio = Math.abs(signedArea) / (width * height);
  const crosses = points.map((point, index) => {
    const next = points[(index + 1) % 4];
    const after = points[(index + 2) % 4];
    return (next.x - point.x) * (after.y - next.y) - (next.y - point.y) * (after.x - next.x);
  });
  const convex = crosses.every((value) => value > 0) || crosses.every((value) => value < 0);
  if (!convex) return { valid: false, areaRatio, message: "四个角点发生交叉，请按指定顺序重新点击" };
  if (areaRatio < 0.015) return { valid: false, areaRatio, message: "标定区域过小，请确认选择了球场外侧四角" };
  const upperAverage = (points[0].y + points[1].y) / 2;
  const lowerAverage = (points[2].y + points[3].y) / 2;
  if (upperAverage >= lowerAverage) return { valid: false, areaRatio, message: "上方角点必须位于下方角点之上" };
  return { valid: true, areaRatio, message: "几何检查通过，可以保存" };
}
