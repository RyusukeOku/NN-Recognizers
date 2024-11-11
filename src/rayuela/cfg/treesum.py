from rayuela.base.semiring import Rational, Real
from rayuela.base.symbol import ε
from rayuela.cfg.nonterminal import NT


class Treesum:
    def __init__(self, cfg):
        self.cfg = cfg

    def sum(self, strategy="forwardchain"):
        table = self.table(strategy)
        # print()
        # print("Table")
        # for k, v in table.items():
        # 	print(k, v)
        # print()
        return table[self.cfg.S]

    def table(self, strategy="forwardchain"):
        if strategy == "forwardchain":
            return self.forwardchain()
        elif strategy == "backwardchain":
            return self.backwardchain()
        elif strategy == "acyclic":
            return self.simpleacyclic()
        else:
            raise NotImplementedError

    def _top_down_step(self, V):
        R = self.cfg.R
        U = R.chart()
        U[self.cfg.S] = R.one

        for p, w in self.cfg.P:
            (head, body) = p
            for X in body:
                U[X] += V[head] * w
        return U

    def _bottom_up_step(self, V):
        R = self.cfg.R

        U = R.chart()
        # ! Added by Anej: Also include the epsilon symbol here.
        # ! I hope this is correct.
        # for a in self.cfg.Sigma:
        for a in self.cfg.Sigma.union({ε}):
            U[a] = R.one

        for p, w in self.cfg.P:
            (head, body) = p
            update = w
            for X in body:
                if isinstance(X, NT):
                    update *= V[X]

            U[head] += update

        return U

    def _judge_of_the_change(self, U, V, tol):
        if self.cfg.R is Real or self.cfg.R is Rational:
            total = 0.0
            for X in self.cfg.V:
                val1, val2 = U[X], V[X]
                total += abs(float(val1) - float(val2))
            if total < tol:
                return True
            return False
        elif self.cfg.R.idempotent:
            for k, v in U.items():
                if v != V[k]:
                    return False
            return True
        else:
            raise NotImplementedError

    def backwardchain(self, tol=1e-100, timeout=10000):
        R = self.cfg.R
        zero, one = R.zero, R.one

        V = R.chart(zero)
        V[self.cfg.S] = one

        counter = 0
        while counter < timeout:
            U = self._top_down_step(V)
            if self._judge_of_the_change(U, V, tol):
                return V
            V = U
            counter += 1

        return V

    def forwardchain(self, tol=1e-100, timeout=100000):
        # Timmy notes that this is naive bottom-up evaluation.
        # semi-naive would be better. See Eisner et al. (2005).
        R = self.cfg.R

        V = R.chart(R.zero)

        counter = 0
        while counter < timeout:
            U = self._bottom_up_step(V)
            if self._judge_of_the_change(U, V, tol):
                return V
            V = U
            counter += 1

        return V

    def simpleacyclic(self):
        """
        Treesum DP algorithms for acyclic cfgs
        """
        cyclic, stack = self.cfg.cyclic()
        assert cyclic == False

        β = self.cfg.R.chart()
        while stack:
            X = stack.pop()
            X_productions = ((p, w) for p, w in self.cfg.P if p[0] == X)
            for p, w in X_productions:
                _, body = p
                update = w
                for elem in body:
                    if isinstance(elem, NT):
                        update *= β[elem]
                β[X] += update

        return β
